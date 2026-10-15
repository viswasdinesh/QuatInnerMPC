############################################################################
# plot_ulog.py
# Post-flight tracking plots for the inner MPC from a PX4 ULog (.ulg) and
# the trajectory .npz the node was given.
#
#   python3 -m px4_mpc.utils.plot_ulog flight.ulg --traj circle.npz \
#       [--out plots_dir] [--settle 5.0]
#
# Requires pyulog (pip install pyulog).
#
# Reference alignment: the node starts its trajectory index on the first
# control step with nav_state == OFFBOARD and armed, then advances one sample
# per 20 ms step. The same instant is taken from vehicle_status here, so the
# reference is accurate to about one control step. The index never resets,
# so only the first OFFBOARD+armed window is plotted.
#
# All quantities are plotted in the MPC frames (ENU world, FLU body), using
# the same conversions as mpc_inner_quadrotor.py.
############################################################################

import argparse
import os

import numpy as np

NAV_STATE_OFFBOARD = 14
ARMING_STATE_ARMED = 2


# ── ULog loading ───────────────────────────────────────────────────────────
def load_ulog(path):
    """Return {topic: {field: array}} with 'timestamp' in seconds (first instance only)."""
    from pyulog import ULog

    topics = ['vehicle_local_position', 'vehicle_attitude', 'vehicle_angular_velocity',
              'vehicle_status', 'vehicle_thrust_setpoint', 'vehicle_torque_setpoint',
              'actuator_motors']
    ulog = ULog(path, topics)
    data = {}
    for d in ulog.data_list:
        if d.multi_id != 0 or d.name in data:
            continue
        fields = dict(d.data)
        fields['timestamp'] = fields['timestamp'] * 1e-6
        data[d.name] = fields
    return data


def offboard_window(status):
    """(t_start, t_end) of the first contiguous OFFBOARD + armed window."""
    t = status['timestamp']
    active = (status['nav_state'] == NAV_STATE_OFFBOARD) & \
             (status['arming_state'] == ARMING_STATE_ARMED)
    if not active.any():
        raise RuntimeError('Log has no OFFBOARD + armed segment.')
    i0 = int(np.argmax(active))
    after = np.nonzero(~active[i0:])[0]
    t_end = t[i0 + after[0]] if len(after) else t[-1]
    return t[i0], t_end


# ── Frame conversions (match mpc_inner_quadrotor.py) ──────────────────────
def ned_to_enu(x, y, z):
    return np.stack([y, x, -z], axis=1)


def quat_ned_frd_to_enu_flu(q):
    qw, qx, qy, qz = q.T
    q_enu = np.stack([qw + qz, qx + qy, qx - qy, qw - qz], axis=1) / np.sqrt(2.0)
    return q_enu / np.linalg.norm(q_enu, axis=1, keepdims=True)


def quat_to_euler(q):
    """[w,x,y,z] → roll, pitch, yaw [rad]."""
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.stack([roll, pitch, yaw], axis=1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp(t_new, t, y):
    y = np.asarray(y)
    if y.ndim == 1:
        return np.interp(t_new, t, y)
    return np.stack([np.interp(t_new, t, y[:, i]) for i in range(y.shape[1])], axis=1)


def cols(fields, name, n):
    return np.stack([fields[f'{name}[{i}]'] for i in range(n)], axis=1)


# ── Build aligned flight arrays ────────────────────────────────────────────
def build_flight(data, traj, dt):
    t0, t1 = offboard_window(data['vehicle_status'])

    lp = data['vehicle_local_position']
    sel = (lp['timestamp'] >= t0) & (lp['timestamp'] <= t1)
    t = lp['timestamp'][sel]                 # common time base
    f = {'t': t - t0, 't_abs': t, 'window': (t0, t1)}
    f['p'] = ned_to_enu(lp['x'], lp['y'], lp['z'])[sel]
    f['v'] = ned_to_enu(lp['vx'], lp['vy'], lp['vz'])[sel]

    att = data['vehicle_attitude']
    q = quat_ned_frd_to_enu_flu(cols(att, 'q', 4))
    f['q'] = interp(t, att['timestamp'], q)
    f['q'] /= np.linalg.norm(f['q'], axis=1, keepdims=True)

    av = data['vehicle_angular_velocity']
    w_frd = cols(av, 'xyz', 3)
    f['w'] = interp(t, av['timestamp'], w_frd * np.array([1.0, -1.0, -1.0]))

    # Reference sample used by the node at each time (index advances every dt, clamped).
    idx = np.clip(np.floor(f['t'] / dt).astype(int), 0, len(traj['p']) - 1)
    for k in ('p', 'v', 'q', 'w'):
        f[k + '_ref'] = traj[k][idx]

    # Setpoints as published (normalised, FRD body), sample-and-hold within window.
    for topic, key in (('vehicle_thrust_setpoint', 'thrust_sp'),
                       ('vehicle_torque_setpoint', 'torque_sp')):
        if topic in data:
            d = data[topic]
            s = (d['timestamp'] >= t0) & (d['timestamp'] <= t1)
            f[key] = (d['timestamp'][s] - t0, cols(d, 'xyz', 3)[s])

    if 'actuator_motors' in data:
        d = data['actuator_motors']
        s = (d['timestamp'] >= t0) & (d['timestamp'] <= t1)
        f['motors'] = (d['timestamp'][s] - t0, cols(d, 'control', 4)[s])
    return f


# ── Metrics ───────────────────────────────────────────────────────────────
def report(f, settle):
    e = f['p'] - f['p_ref']
    s = f['t'] >= settle
    print(f'OFFBOARD window: {f["t"][-1]:.1f} s ({len(f["t"])} samples), '
          f'metrics after t = {settle:.1f} s:')
    if not s.any():
        print('  (window shorter than settle time)')
        return
    rms = np.sqrt(np.mean(e[s]**2, axis=0))
    print(f'  position RMS  x {rms[0]:.3f}  y {rms[1]:.3f}  z {rms[2]:.3f}  '
          f'|e| {np.sqrt(np.mean(np.sum(e[s]**2, axis=1))):.3f} m')
    print(f'  position max  |e| {np.linalg.norm(e[s], axis=1).max():.3f} m, '
          f'mean z error {e[s, 2].mean():+.3f} m')
    ev = f['v'] - f['v_ref']
    print(f'  velocity RMS  |e| {np.sqrt(np.mean(np.sum(ev[s]**2, axis=1))):.3f} m/s')
    yaw_e = wrap(quat_to_euler(f['q'])[:, 2] - quat_to_euler(f['q_ref'])[:, 2])
    print(f'  yaw RMS {np.degrees(np.sqrt(np.mean(yaw_e[s]**2))):.2f} deg')
    if 'thrust_sp' in f:
        thr = -f['thrust_sp'][1][:, 2]
        print(f'  thrust_sp  mean {thr.mean():.3f}  max {thr.max():.3f}  '
              f'(saturated {100 * np.mean(thr >= 0.999):.1f}%)')
    if 'torque_sp' in f:
        tq = np.abs(f['torque_sp'][1])
        print(f'  |torque_sp| max  x {tq[:, 0].max():.3f}  y {tq[:, 1].max():.3f}  '
              f'z {tq[:, 2].max():.3f}  (saturated {100 * np.mean(tq >= 0.999):.1f}%)')
    if 'motors' in f:
        m = f['motors'][1]
        print(f'  motors  max {np.nanmax(m):.3f}  min {np.nanmin(m):.3f}  '
              f'(any ≥ 0.99: {100 * np.mean(np.any(m >= 0.99, axis=1)):.1f}% of samples)')


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_all(f, out_dir, settle):
    import matplotlib
    matplotlib.use('Agg')   # headless (Pi / SSH)
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t = f['t']
    saved = []

    def finish(fig, name):
        for a in fig.axes:
            if a.name == '3d':
                continue
            a.minorticks_on()
            a.grid(True, which='major', alpha=0.45)
            a.grid(True, which='minor', alpha=0.15, ls=':')
        fig.tight_layout()
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    def settle_line(a):
        a.axvline(settle, color='gray', ls=':', lw=1)

    # 1. Top view + 3D
    fig = plt.figure(figsize=(12, 5.5))
    a = fig.add_subplot(1, 2, 1)
    a.plot(f['p_ref'][:, 0], f['p_ref'][:, 1], '--', color='k', lw=1.0, zorder=3, label='reference')
    a.plot(f['p'][:, 0], f['p'][:, 1], label='actual')
    a.plot(*f['p'][0, :2], 'go', label='offboard start')
    a.set(title='Top view (ENU)', xlabel='x [m]', ylabel='y [m]', aspect='equal')
    a.legend()
    a = fig.add_subplot(1, 2, 2, projection='3d')
    a.plot(*f['p_ref'].T, '--', color='k', lw=1.0, zorder=3)
    a.plot(*f['p'].T)
    a.set(title='3D path', xlabel='x [m]', ylabel='y [m]', zlabel='z [m]')
    finish(fig, '01_path.png')

    # 2. Position and position error
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    e = f['p'] - f['p_ref']
    for i, n in enumerate('xyz'):
        ax[i].plot(t, f['p_ref'][:, i], '--', color='k', lw=1.0, zorder=3, label='reference')
        ax[i].plot(t, f['p'][:, i], label='actual')
        ax[i].set_ylabel(f'{n} [m]')
        settle_line(ax[i])
    ax[0].legend(loc='upper right')
    for i, n in enumerate('xyz'):
        ax[3].plot(t, e[:, i], label=f'e_{n}')
    ax[3].plot(t, np.linalg.norm(e, axis=1), 'k', lw=1, label='|e|')
    s = t >= settle
    if s.any():
        # Scale to the settled part so steady-state error is readable.
        lim = 1.3 * np.linalg.norm(e[s], axis=1).max()
        ax[3].set_ylim(-lim, lim)
    ax[3].set(ylabel='error [m]', xlabel='t since offboard [s]')
    ax[3].legend(loc='upper right', ncol=4)
    settle_line(ax[3])
    ax[0].set_title('Position tracking (ENU)')
    finish(fig, '02_position.png')

    # 3. Velocity
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, n in enumerate('xyz'):
        ax[i].plot(t, f['v_ref'][:, i], '--', color='k', lw=1.0, zorder=3, label='reference')
        ax[i].plot(t, f['v'][:, i], label='actual')
        ax[i].set_ylabel(f'v{n} [m/s]')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Velocity tracking (ENU)')
    ax[-1].set_xlabel('t since offboard [s]')
    finish(fig, '03_velocity.png')

    # 4. Attitude
    eul = np.degrees(quat_to_euler(f['q']))
    eul_ref = np.degrees(quat_to_euler(f['q_ref']))
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for i, n in enumerate(['roll', 'pitch']):
        ax[i].plot(t, eul_ref[:, i], '--', color='k', lw=1.0, zorder=3, label='reference')
        ax[i].plot(t, eul[:, i], label='actual')
        ax[i].set_ylabel(f'{n} [deg]')
    ax[2].plot(t, np.degrees(np.unwrap(np.radians(eul_ref[:, 2]))), '--', color='k', lw=1.0, zorder=3)
    ax[2].plot(t, np.degrees(np.unwrap(np.radians(eul[:, 2]))))
    ax[2].set_ylabel('yaw (unwrapped) [deg]')
    ax[3].plot(t, np.degrees(wrap(np.radians(eul[:, 2] - eul_ref[:, 2]))), label='yaw error')
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (f['q'][:, 1]**2 + f['q'][:, 2]**2), -1, 1)))
    ax[3].plot(t, tilt, label='tilt')
    ax[3].set(ylabel='[deg]', xlabel='t since offboard [s]')
    ax[3].legend(loc='upper right')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Attitude (ENU / FLU, ZYX Euler)')
    finish(fig, '04_attitude.png')

    # 5. Body rates
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, n in enumerate('xyz'):
        ax[i].plot(t, f['w_ref'][:, i], '--', color='k', lw=1.0, zorder=3, label='reference')
        ax[i].plot(t, f['w'][:, i], label='actual', lw=0.8)
        ax[i].set_ylabel(f'ω{n} [rad/s]')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Body rates (FLU)')
    ax[-1].set_xlabel('t since offboard [s]')
    finish(fig, '05_body_rates.png')

    # 6. Actuation: published setpoints and motor outputs
    keys = [k for k in ('thrust_sp', 'torque_sp', 'motors') if k in f]
    if keys:
        fig, ax = plt.subplots(len(keys), 1, figsize=(11, 3 * len(keys) + 1),
                               sharex=True, squeeze=False)
        ax = ax[:, 0]
        for a, k in zip(ax, keys):
            ts, y = f[k]
            if k == 'thrust_sp':
                a.plot(ts, -y[:, 2], label='thrust (−z, FRD)')
                a.axhline(1.0, color='r', ls='--', lw=1)
                a.set(ylabel='normalised', title='Thrust setpoint')
            elif k == 'torque_sp':
                for i, n in enumerate('xyz'):
                    a.plot(ts, y[:, i], lw=0.8, label=f'τ{n} (FRD)')
                for lim in (-1, 1):
                    a.axhline(lim, color='r', ls='--', lw=1)
                a.set(ylabel='normalised', title='Torque setpoint')
            else:
                for i in range(4):
                    a.plot(ts, y[:, i], lw=0.8, label=f'motor {i + 1}')
                a.axhline(1.0, color='r', ls='--', lw=1)
                a.set(ylabel='[0, 1]', title='actuator_motors')
            a.legend(loc='upper right', ncol=4)
        ax[-1].set_xlabel('t since offboard [s]')
        finish(fig, '06_actuation.png')

    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ulog', help='PX4 .ulg file')
    parser.add_argument('--traj', required=True, help='trajectory .npz given to the node')
    parser.add_argument('--out', default='', help='output dir (default: <ulog>_plots)')
    parser.add_argument('--settle', type=float, default=5.0,
                        help='seconds after offboard start excluded from metrics')
    parser.add_argument('--dt', type=float, default=0.02, help='node control period [s]')
    args = parser.parse_args()

    traj = dict(np.load(args.traj))
    f = build_flight(load_ulog(args.ulog), traj, args.dt)
    report(f, args.settle)
    out = args.out or os.path.splitext(args.ulog)[0] + '_plots'
    for p in plot_all(f, out, args.settle):
        print(f'  saved {p}')


if __name__ == '__main__':
    main()
