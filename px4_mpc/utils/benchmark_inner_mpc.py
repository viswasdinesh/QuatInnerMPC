############################################################################
# benchmark_inner_mpc.py
# Closed-loop timing benchmark for MultirotorInnerMPC (desktop or Pi 5).
#
# Tracks a horizontal circle at z = 3 m, closing the loop with the acados
# integrator, and reports per-solve timing against the 20 ms (50 Hz) budget.
#
#   python3 -m px4_mpc.utils.benchmark_inner_mpc [--steps 1500] [--save out.npz]
#       [--plot tracking.png]
#
# Timing is reported separately for the first WARMUP_STEPS solves (cold
# caches, page faults) and for the steady state after them.
#
# On a Pi 5, set the CPU governor to "performance" first so the numbers
# are not dominated by frequency scaling.
############################################################################

import argparse
import platform
import time

import numpy as np

from px4_mpc.models.multirotor_inner_model import MultirotorInnerModel
from px4_mpc.controllers.multirotor_inner_mpc import MultirotorInnerMPC

WARMUP_STEPS = 50


def circle_reference(t, radius=1.5, period=6.0, z=3.0):
    """Stage parameters [p_ref, v_ref, q_ref, w_ref] for a level circle."""
    w = 2.0 * np.pi / period
    p = np.array([radius * np.cos(w * t), radius * np.sin(w * t), z])
    v = np.array([-radius * w * np.sin(w * t), radius * w * np.cos(w * t), 0.0])
    return np.concatenate([p, v, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def stats(name, ms, budget_ms):
    print(f'  {name:<14} mean {ms.mean():7.3f}  p50 {np.percentile(ms, 50):7.3f}  '
          f'p99 {np.percentile(ms, 99):7.3f}  max {ms.max():7.3f} ms  '
          f'({100.0 * ms.mean() / budget_ms:4.1f}% of budget)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--plot', type=str, default='inner_mpc_tracking.png',
                        help='tracking plot output path (empty to skip)')
    args = parser.parse_args()

    model = MultirotorInnerModel()
    t0 = time.perf_counter()
    mpc = MultirotorInnerMPC(model)
    build_s = time.perf_counter() - t0
    solver, integrator = mpc.ocp_solver, mpc.integrator

    N, dt = mpc.N, mpc.Tf / mpc.N
    budget_ms = dt * 1e3
    u_hover = np.array([model.mass * 9.81, 0.0, 0.0, 0.0])

    # Start hovering 0.5 m below and 1.5 m off the circle's start point.
    x = np.array([0.0, 0.0, 2.5,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0])
    for k in range(N + 1):
        solver.set(k, 'x', x)
    for k in range(N):
        solver.set(k, 'u', u_hover)

    K = args.steps
    t_tot, t_lin, t_qp, t_wall = (np.zeros(K) for _ in range(4))
    X, U, R = np.zeros((K + 1, 13)), np.zeros((K, 4)), np.zeros((K, 13))
    X[0] = x
    failures = 0

    for i in range(K):
        for k in range(N + 1):
            solver.set(k, 'p', circle_reference((i + k) * dt))
        R[i] = circle_reference(i * dt)

        # Wall time includes the Python get/set overhead the ROS node pays.
        w0 = time.perf_counter()
        try:
            simU, _ = mpc.solve(x)
            u = simU[0]
        except Exception:
            failures += 1
            u = U[i - 1] if i > 0 else u_hover
        t_wall[i] = (time.perf_counter() - w0) * 1e3

        t_tot[i] = solver.get_stats('time_tot') * 1e3
        t_lin[i] = solver.get_stats('time_lin') * 1e3
        t_qp[i] = solver.get_stats('time_qp') * 1e3

        integrator.set('p', R[i])
        x = integrator.simulate(x=x, u=u)
        x[6:10] /= np.linalg.norm(x[6:10])
        X[i + 1], U[i] = x, u

    pos_err = np.linalg.norm(X[1:, 0:3] - R[:, 0:3], axis=1)
    settled = pos_err[int(5.0 / dt):]   # skip the first 5 s of convergence

    print(f'\nMultirotorInnerMPC benchmark: {platform.machine()} {platform.processor()}')
    print(f'  N = {N}, Tf = {mpc.Tf} s, {K} steps, build {build_s:.1f} s, '
          f'failed solves {failures}')
    for label, seg in ((f'first {WARMUP_STEPS} steps (warm-up)', slice(0, WARMUP_STEPS)),
                       (f'steps {WARMUP_STEPS}-{K} (steady state)', slice(WARMUP_STEPS, K))):
        if len(t_tot[seg]) == 0:
            continue
        print(f'  -- {label}')
        stats('acados total', t_tot[seg], budget_ms)
        stats('  linearize', t_lin[seg], budget_ms)
        stats('  QP', t_qp[seg], budget_ms)
        stats('solve() wall', t_wall[seg], budget_ms)
    print(f'  tracking RMS after 5 s: {np.sqrt(np.mean(settled**2)):.4f} m, '
          f'max {settled.max():.4f} m')

    if args.save:
        np.savez(args.save, t_tot=t_tot, t_lin=t_lin, t_qp=t_qp, t_wall=t_wall,
                 X=X, U=U, R=R)
        print(f'  saved {args.save}')

    if args.plot:
        plot_tracking(args.plot, dt, X, R, pos_err)
        print(f'  saved {args.plot}')


def plot_tracking(path, dt, X, R, pos_err, settle=5.0):
    import matplotlib
    matplotlib.use('Agg')   # headless (Pi / SSH)
    import matplotlib.pyplot as plt

    t = np.arange(len(R)) * dt
    P = X[1:, 0:3]
    e = P - R[:, 0:3]
    s = t >= settle

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.7, 1])

    # Left column: one axis per coordinate so each gets its own scale,
    # reference drawn on top so it stays visible where the curves overlap.
    ax_p = [fig.add_subplot(gs[0, 0])]
    ax_p += [fig.add_subplot(gs[i, 0], sharex=ax_p[0]) for i in (1, 2)]
    for i, (a, name) in enumerate(zip(ax_p, 'xyz')):
        a.plot(t, P[:, i], color='C0', lw=1.6, label='actual')
        a.plot(t, R[:, i], '--', color='k', lw=1.0, label='reference', zorder=3)
        a.axvline(settle, color='gray', ls=':', lw=1)
        a.set_ylabel(f'{name} [m]')
    ax_p[0].legend(loc='upper right', ncol=2)
    ax_p[0].set_title('Position: actual vs reference')

    # Error axis scaled to the settled part; the start-up transient is clipped.
    ax_e = fig.add_subplot(gs[3, 0], sharex=ax_p[0])
    for i, name in enumerate('xyz'):
        ax_e.plot(t, e[:, i], lw=1.2, label=f'e_{name}')
    ax_e.plot(t, pos_err, 'k', lw=1.0, label='|e|')
    ax_e.axhline(0.0, color='gray', lw=0.8)
    ax_e.axvline(settle, color='gray', ls=':', lw=1)
    if s.any():
        lim = 1.3 * max(np.abs(e[s]).max(), pos_err[s].max())
        ax_e.set_ylim(-lim, lim)
    ax_e.set(xlabel='t [s]', ylabel='error [m]',
             title=f'Position error (scaled to t ≥ {settle:g} s, start-up transient clipped)')
    ax_e.legend(loc='upper right', ncol=4)

    top = fig.add_subplot(gs[0:2, 1])
    top.plot(P[:, 0], P[:, 1], color='C0', lw=1.6, label='actual')
    top.plot(R[:, 0], R[:, 1], '--', color='k', lw=1.0, label='reference', zorder=3)
    top.plot(*P[0, :2], 'go', label='start')
    top.set(title='Top view', xlabel='x [m]', ylabel='y [m]', aspect='equal')
    top.legend(loc='upper right')

    # Tilt angle from the quaternion: angle between body z and world z.
    q = X[1:, 6:10]
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[:, 1]**2 + q[:, 2]**2), -1.0, 1.0)))
    ax_t = fig.add_subplot(gs[2:4, 1])
    ax_t.plot(t, tilt)
    ax_t.set(title='Tilt angle', xlabel='t [s]', ylabel='[deg]')

    for a in fig.axes:
        a.minorticks_on()
        a.grid(True, which='major', alpha=0.45)
        a.grid(True, which='minor', alpha=0.15, ls=':')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
