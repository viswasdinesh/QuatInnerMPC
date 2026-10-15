############################################################################
# plot_node_log.py
# Plots and summary for a run log written by mpc_inner_quadrotor
# (utils/run_logger.py), complementing the PX4 ULog plots:
#
#   python3 -m px4_mpc.utils.plot_node_log ~/mpc_logs/<run>.npz [--out dir]
#
#   01_timing.png      solve time, callback→publish latency, input ages, failures
#   02_commands.png    raw MPC thrust / torques vs limits, clipping
#   03_tracking.png    position, attitude and rates the MPC saw vs its reference
#   04_prediction.png  what the MPC predicted vs what was measured j steps later
############################################################################

import argparse
import os

import numpy as np

from px4_mpc.utils.plot_ulog import quat_to_euler, wrap


def load(path):
    d = dict(np.load(path, allow_pickle=True))
    d['meta'] = d['meta'].item()
    return d


def window_ids(d):
    """Window number for every row."""
    ids = np.zeros(len(d['t_loop']), dtype=int)
    for i, s in enumerate(d['window_start']):
        ids[s:] = i
    return ids


def prediction_error(d, j):
    """Measured minus predicted state, j steps ahead (rows where both exist in one window)."""
    n = len(d['t_loop'])
    win = window_ids(d)
    i = np.arange(n - j)
    ok = (win[i] == win[i + j]) & np.isfinite(d['x_pred'][i, j]).all(axis=1) \
         & np.isfinite(d['x0'][i + j]).all(axis=1)
    i = i[ok]
    return i, d['x0'][i + j] - d['x_pred'][i, j].astype(np.float64)


def report(d):
    m, dt = d['meta'], d['meta']['dt']
    n = len(d['t_loop'])
    ok = d['status'] == 0
    lat = (d['t_pub'] - d['t_loop'])[ok] * 1e3
    print(f'{n} control steps in {len(d["window_start"])} OFFBOARD window(s), '
          f'trajectory: {m.get("trajectory_path") or "default hover"}')
    print(f'  failed solves {np.sum(~ok)} '
          f'(statuses {sorted(set(d["status"][~ok].astype(int).tolist()))})')
    st = d['solve_time'][np.isfinite(d['solve_time'])] * 1e3
    print(f'  solve time  p50 {np.median(st):.2f}  p99 {np.percentile(st, 99):.2f}  max {st.max():.2f} ms')
    print(f'  callback→publish  p50 {np.median(lat):.2f}  p99 {np.percentile(lat, 99):.2f}  '
          f'max {lat.max():.2f} ms  (over {dt * 1e3:.0f} ms budget: {np.sum(lat > dt * 1e3)})')
    gaps = np.diff(d['t_loop']) * 1e3
    same = window_ids(d)[1:] == window_ids(d)[:-1]
    print(f'  loop period p50 {np.median(gaps[same]):.1f}  max {gaps[same].max():.1f} ms')
    for k in ('att', 'angvel', 'lpos'):
        a = d[f'age_{k}'] * 1e3
        print(f'  input age {k:<6} p50 {np.nanmedian(a):5.1f}  p95 {np.nanpercentile(a, 95):5.1f}  '
              f'max {np.nanmax(a):6.1f} ms')
    tu = np.abs(d['torque_unclipped'][ok])
    print('  torque beyond ±1 (clipped): ' + '  '.join(
        f'{n_} {100 * np.mean(tu[:, i] > 1):.1f}%' for i, n_ in enumerate(('x', 'y', 'z'))) +
        f'   (max |unclipped| {np.nanmax(tu, axis=0).round(2).tolist()})')
    for j in (1, 5):
        _, e = prediction_error(d, j)
        if len(e):
            rms = np.sqrt(np.mean(e**2, axis=0))
            print(f'  prediction error {j} step(s) ahead: pos {np.linalg.norm(rms[0:3]):.3f} m, '
                  f'vel {np.linalg.norm(rms[3:6]):.3f} m/s, '
                  f'rates x/y/z {rms[10]:.2f} / {rms[11]:.2f} / {rms[12]:.2f} rad/s')
    if 'W_diag' in m:
        print(f'  W diag {m["W_diag"]}')


def plot_all(d, out_dir):
    import matplotlib
    matplotlib.use('Agg')   # headless (Pi / SSH)
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    m, dt = d['meta'], d['meta']['dt']
    t = d['t_loop'] - d['t_loop'][0]
    ok = d['status'] == 0
    saved = []

    def finish(fig, name):
        for a in fig.axes:
            a.minorticks_on()
            a.grid(True, which='major', alpha=0.45)
            a.grid(True, which='minor', alpha=0.15, ls=':')
        fig.tight_layout()
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    def marks(a):
        for s in d['window_start'][1:]:
            a.axvline(t[s], color='gray', ls='--', lw=1)
        for tf in t[~ok]:
            a.axvline(tf, color='r', alpha=0.4, lw=1)

    # 1. Timing
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, d['solve_time'] * 1e3, lw=0.8, label='acados solve')
    ax[0].plot(t, (d['t_pub'] - d['t_loop']) * 1e3, lw=0.8, label='callback → publish')
    ax[0].axhline(dt * 1e3, color='r', ls='--', lw=1, label='budget')
    ax[0].set(ylabel='[ms]', title='Timing (red lines = failed solves, dashed = new window)')
    ax[0].legend(loc='upper right', ncol=3)
    gaps = np.r_[np.nan, np.diff(d['t_loop'])] * 1e3
    gaps[d['window_start']] = np.nan
    ax[1].plot(t, gaps, lw=0.8)
    ax[1].axhline(dt * 1e3, color='gray', ls=':', lw=1)
    ax[1].set(ylabel='loop period [ms]')
    for k in ('att', 'angvel', 'lpos'):
        ax[2].plot(t, d[f'age_{k}'] * 1e3, lw=0.8, label=k)
    ax[2].set(ylabel='input age [ms]', xlabel='t [s]')
    ax[2].legend(loc='upper right', ncol=3)
    for a in ax:
        marks(a)
    finish(fig, '01_timing.png')

    # 2. Commands
    tau_max = np.asarray(m['tau_max'])
    fig, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    ax[0].plot(t, d['u_raw'][:, 0], lw=0.9)
    ax[0].axhline(m['mass'] * 9.81, color='gray', ls=':', lw=1, label='hover')
    ax[0].axhline(m['F_max'], color='r', ls='--', lw=1, label='F_max')
    ax[0].set(ylabel='F [N]', title='MPC output before normalisation (FLU)')
    ax[0].legend(loc='upper right')
    for i, n in enumerate('xyz'):
        a = ax[i + 1]
        a.plot(t, d['u_raw'][:, i + 1], lw=0.9, label=f'τ{n} raw [Nm]')
        for s in (-1, 1):
            a.axhline(s * tau_max[i], color='r', ls='--', lw=1)
        a.set_ylabel(f'τ{n} [Nm]')
        marks(a)
    ax[4].plot(t, d['torque_unclipped'][:, 1], lw=0.9, label='τy / tau_max (FRD, unclipped)')
    ax[4].plot(t, d['torque_norm'][:, 1], lw=0.9, label='τy published (clipped)')
    for s in (-1, 1):
        ax[4].axhline(s, color='r', ls='--', lw=1)
    ax[4].set(ylabel='normalised', xlabel='t [s]')
    ax[4].legend(loc='upper right')
    marks(ax[0])
    finish(fig, '02_commands.png')

    # 3. Tracking as seen by the MPC
    x, r = d['x0'], d['x_ref']
    eul, eul_r = np.degrees(quat_to_euler(x[:, 6:10])), np.degrees(quat_to_euler(r[:, 6:10]))
    fig, ax = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for i, n in enumerate('xyz'):
        ax[i, 0].plot(t, x[:, i], lw=1.4, label='measured')
        ax[i, 0].plot(t, r[:, i], '--', color='k', lw=1.0, zorder=3, label='reference')
        ax[i, 0].set_ylabel(f'{n} [m]')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_title('Position (state given to the MPC)')
    for i, n in enumerate(('roll', 'pitch')):
        ax[i, 1].plot(t, eul[:, i], lw=1.2)
        ax[i, 1].plot(t, eul_r[:, i], '--', color='k', lw=1.0, zorder=3)
        ax[i, 1].set_ylabel(f'{n} [deg]')
    ax[0, 1].set_title('Attitude and rates (FLU)')
    for i, n in enumerate('xyz'):
        ax[2, 1].plot(t, x[:, 10 + i], lw=0.8, label=f'ω{n}')
    ax[2, 1].set_ylabel('[rad/s]')
    ax[2, 1].legend(loc='upper right', ncol=3)
    ax[2, 0].set_xlabel('t [s]')
    ax[2, 1].set_xlabel('t [s]')
    for a in ax.flat:
        marks(a)
    finish(fig, '03_tracking.png')

    # 4. Prediction vs measurement (model mismatch): pitch rate and z
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, x[:, 11], color='k', lw=1.2, label='measured ωy')
    for j in (1, 3, 5):
        i, _ = prediction_error(d, j)
        ax[0].plot(t[i + j], d['x_pred'][i, j, 11], lw=0.8, label=f'predicted {j} step(s) earlier')
    ax[0].set(ylabel='ωy [rad/s]', title='Pitch rate: measured vs what the MPC predicted')
    ax[0].legend(loc='upper right', ncol=2)
    jj = np.arange(1, m['N'] + 1)
    rms = np.array([np.sqrt(np.mean(prediction_error(d, j)[1][:, 10:13]**2, axis=0)) for j in jj])
    ax[1].remove()
    ax[1] = fig.add_subplot(3, 1, 2)
    for k, n in enumerate('xyz'):
        ax[1].plot(jj * dt * 1e3, rms[:, k], 'o-', ms=3, label=f'ω{n}')
    ax[1].set(xlabel='prediction horizon [ms]', ylabel='RMS error [rad/s]',
              title='Body-rate prediction error vs horizon')
    ax[1].legend(loc='upper left')
    i, e = prediction_error(d, 5)
    ax[2].plot(t[i + 5], e[:, 2], lw=0.9, label='z error, 5 steps ahead [m]')
    ax[2].plot(t[i + 5], e[:, 5], lw=0.9, label='vz error, 5 steps ahead [m/s]')
    ax[2].set(xlabel='t [s]', ylabel='measured − predicted')
    ax[2].legend(loc='upper right')
    marks(ax[0])
    marks(ax[2])
    finish(fig, '04_prediction.png')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', help='run log .npz from mpc_inner_quadrotor')
    parser.add_argument('--out', default='', help='output dir (default: <log>_plots)')
    args = parser.parse_args()

    d = load(args.log)
    report(d)
    out = args.out or os.path.splitext(args.log)[0] + '_plots'
    for p in plot_all(d, out):
        print(f'  saved {p}')


if __name__ == '__main__':
    main()
