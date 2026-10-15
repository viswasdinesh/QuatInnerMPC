############################################################################
# trajectory.py
# Utility functions to generate and save pre-computed reference trajectories.
#
# Usage (offline, run once before launching the node):
#
#   from px4_mpc.utils.trajectory import generate_hover_trajectory
#   generate_hover_trajectory(60.0, 0.02, [0., 0., 3.],
#                             '/path/to/trajectory.npz')
#
# The saved .npz file contains arrays:
#   t  (M,)    time stamps [s]
#   p  (M, 3)  position    [m]     ENU
#   v  (M, 3)  velocity    [m/s]   ENU
#   q  (M, 4)  quaternion  [qw,qx,qy,qz]  ENU→FLU
#   w  (M, 3)  body rates  [rad/s] FLU
#
# Sampling rate must match MPC dt = Tf/N (default 0.02 s = 50 Hz).
############################################################################

import numpy as np
import os


def _save(path: str, t, p, v, q, w):
    """Save trajectory arrays to a .npz file and print a summary."""
    np.savez(path, t=t, p=p, v=v, q=q, w=w)
    duration = t[-1] - t[0]
    print(f'Saved trajectory → {os.path.abspath(path)}')
    print(f'  Steps: {len(t)}   Duration: {duration:.1f} s   dt: {t[1]-t[0]:.4f} s')


def generate_hover_trajectory(t_total: float, dt: float,
                               p_target, save_path: str):
    """
    Static hover: the drone takes off to p_target and holds.
    All velocity, rate, and attitude references are zero / identity.

    Args:
        t_total:    total trajectory duration [s]
        dt:         sample interval [s]  — must match MPC dt
        p_target:   [x, y, z] target position in ENU [m]
        save_path:  absolute path for output .npz file
    """
    p_target = np.asarray(p_target, dtype=np.float64)
    M = int(t_total / dt)
    t = np.arange(M) * dt

    p = np.tile(p_target,        (M, 1))
    v = np.zeros((M, 3))
    q = np.tile([1., 0., 0., 0.], (M, 1))   # level, no yaw
    w = np.zeros((M, 3))

    _save(save_path, t, p, v, q, w)


def generate_takeoff_trajectory(t_total: float, dt: float,
                                 p_start, p_target,
                                 climb_duration: float,
                                 save_path: str):
    """
    Smooth takeoff: minimum-jerk (quintic) climb from p_start to p_target
    over `climb_duration` seconds, then a static hover at p_target for the
    remainder of t_total.

    Unlike generate_hover_trajectory(), the velocity reference here actually
    matches the commanded motion during the climb (v_ref != 0 while p_ref is
    moving), so the position-tracking cost and the velocity-tracking cost
    pull in the same direction instead of fighting each other. The quintic
    profile has zero velocity AND zero acceleration at both the start and
    end of the climb segment, so there's no discontinuity where it splices
    into the hold phase.

    Args:
        t_total:        total trajectory duration [s]
        dt:             sample interval [s]  — must match MPC dt
        p_start:        [x, y, z] starting position in ENU [m]
                         (should match the vehicle's actual takeoff point)
        p_target:       [x, y, z] hover target position in ENU [m]
        climb_duration: time to go from p_start to p_target [s]
        save_path:      absolute path for output .npz file
    """
    p_start  = np.asarray(p_start,  dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)

    M = int(t_total / dt)
    t = np.arange(M) * dt

    # Normalised time within the climb segment, clipped to [0, 1] so the
    # profile freezes at the fully-arrived state once climb_duration elapses.
    tau = np.clip(t / climb_duration, 0.0, 1.0)

    # Quintic minimum-jerk scalar profile: s(0)=0, s(1)=1,
    # s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    s      = 6.0*tau**5 - 15.0*tau**4 + 10.0*tau**3
    s_dot  = (30.0*tau**4 - 60.0*tau**3 + 30.0*tau**2) / climb_duration

    # After the climb segment ends, tau is pinned at 1.0, so s=1 and
    # s_dot=0 automatically — i.e. a clean static hold, no extra masking
    # needed.
    delta = p_target - p_start
    p = p_start[None, :] + delta[None, :] * s[:, None]
    v = delta[None, :] * s_dot[:, None]

    q = np.tile([1., 0., 0., 0.], (M, 1))   # level, no yaw
    w = np.zeros((M, 3))

    _save(save_path, t, p, v, q, w)


def generate_circle_trajectory(t_total: float, dt: float,
                                radius: float, height: float,
                                freq: float, save_path: str):
    """
    Horizontal circle at constant height, heading aligned to velocity.

    Args:
        t_total:    total trajectory duration [s]
        dt:         sample interval [s]  — must match MPC dt
        radius:     circle radius [m]
        height:     flight height in ENU z [m]
        freq:       circle frequency [Hz]  (e.g. 0.1 → 10 s per lap)
        save_path:  absolute path for output .npz file
    """
    M     = int(t_total / dt)
    t     = np.arange(M) * dt
    omega = 2.0 * np.pi * freq

    p = np.stack([
        radius * np.cos(omega * t),
        radius * np.sin(omega * t),
        np.full(M, height),
    ], axis=1)   # (M, 3)

    v = np.stack([
        -radius * omega * np.sin(omega * t),
         radius * omega * np.cos(omega * t),
         np.zeros(M),
    ], axis=1)   # (M, 3)

    # Heading = tangent direction (yaw only, no roll/pitch)
    yaw = np.arctan2(v[:, 1], v[:, 0])
    q = np.stack([
        np.cos(yaw / 2),
        np.zeros(M),
        np.zeros(M),
        np.sin(yaw / 2),
    ], axis=1)   # (M, 4)  [qw, qx, qy, qz]

    # Yaw rate = dω/dt = omega (constant angular speed on circle)
    w = np.zeros((M, 3))
    w[:, 2] = omega   # body z-rate in FLU

    _save(save_path, t, p, v, q, w)


def generate_figure8_trajectory(t_total: float, dt: float,
                                 scale: float, height: float,
                                 freq: float, save_path: str):
    """
    Lemniscate of Bernoulli (figure-8) in the ENU x-y plane.

    Parametric:
        x(t) = scale * sin(ωt)          * normalisation
        y(t) = scale * sin(ωt)*cos(ωt)  * normalisation

    Args:
        t_total:    total trajectory duration [s]
        dt:         sample interval [s]
        scale:      half-width of the figure-8 [m]
        height:     constant ENU z [m]
        freq:       loop frequency [Hz]
        save_path:  absolute path for output .npz file
    """
    M     = int(t_total / dt)
    t     = np.arange(M) * dt
    omega = 2.0 * np.pi * freq

    # Lemniscate parametric equations
    denom = 1.0 + np.cos(omega * t)**2   # avoid divide-by-zero (always ≥ 1)
    x =  scale * np.sin(omega * t) / denom
    y =  scale * np.sin(omega * t) * np.cos(omega * t) / denom

    # Analytical time derivatives
    ot   = omega * t
    d    = 1.0 + np.cos(ot)**2
    d2   = d**2

    dx = scale * omega * (np.cos(ot) * d + 2.0 * np.sin(ot)**2 * np.cos(ot)) / d2
    dy = scale * omega * (
        (np.cos(2*ot) * d + 2.0 * np.sin(ot)*np.cos(ot)**2 * 2.0*np.sin(ot)) / d2
    )

    p = np.stack([x, y, np.full(M, height)], axis=1)
    v = np.stack([dx, dy, np.zeros(M)],      axis=1)

    yaw = np.arctan2(v[:, 1], v[:, 0])
    q   = np.stack([
        np.cos(yaw / 2),
        np.zeros(M),
        np.zeros(M),
        np.sin(yaw / 2),
    ], axis=1)

    # Approximate yaw rate via finite difference
    yaw_rate    = np.gradient(yaw, dt)
    w           = np.zeros((M, 3))
    w[:, 2]     = yaw_rate

    _save(save_path, t, p, v, q, w)


def generate_bspline_trajectory(waypoints, duration: float, dt: float,
                                yaw: float, save_path: str,
                                hold_start: float = 2.0,
                                hold_end: float = 5.0,
                                g: float = 9.81):
    """
    Quintic interpolating B-spline through `waypoints`, at rest at both ends.

    Zero velocity AND acceleration are imposed at the first and last
    waypoint, so the reference starts exactly at hover (no step on
    hand-over). Knot times are spaced by chord length.

    Attitude/rate references are derived by differential flatness: the body
    z-axis follows (a + g), heading is held at the fixed `yaw`, and the body
    rates come from ω = 2·vec(q⁻¹ ⊗ q̇). This keeps q_ref / w_ref consistent
    with p_ref instead of asking for a level attitude while accelerating.

    Args:
        waypoints:   (K, 3) ENU positions [m], K >= 2
        duration:    time to traverse the spline [s]
        dt:          sample interval [s]  — must match MPC dt
        yaw:         fixed ENU heading [rad] (0 = east, CCW positive)
        save_path:   absolute path for output .npz file
        hold_start:  hover time at the first waypoint before moving [s]
        hold_end:    hover time at the last waypoint after arriving [s]
    """
    from scipy.interpolate import make_interp_spline

    wp = np.asarray(waypoints, dtype=np.float64)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    knots_t = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum() * duration

    rest = [(1, np.zeros(3)), (2, np.zeros(3))]
    spl = make_interp_spline(knots_t, wp, k=5, bc_type=(rest, rest))

    M = int((hold_start + duration + hold_end) / dt)
    t = np.arange(M) * dt
    s = np.clip(t - hold_start, 0.0, duration)
    moving = ((t >= hold_start) & (t <= hold_start + duration))[:, None]

    p = spl(s)
    v = spl(s, 1) * moving
    a = spl(s, 2) * moving

    # Flatness: thrust direction + fixed heading → rotation matrix → quaternion
    z_b = a + np.array([0.0, 0.0, g])
    z_b /= np.linalg.norm(z_b, axis=1, keepdims=True)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b /= np.linalg.norm(y_b, axis=1, keepdims=True)
    x_b = np.cross(y_b, z_b)
    R = np.stack([x_b, y_b, z_b], axis=2)   # columns = body axes in ENU

    q = np.empty((M, 4))
    q[:, 0] = 0.5 * np.sqrt(np.maximum(1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], 1e-12))
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * q[:, 0])
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * q[:, 0])
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * q[:, 0])
    # Keep consecutive samples in the same hemisphere so q̇ is continuous
    for i in range(1, M):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]

    # Body rates: ω = 2 · vec(conj(q) ⊗ q̇)
    qd = np.gradient(q, dt, axis=0)
    qw, qv = q[:, 0], q[:, 1:]
    dw, dv = qd[:, 0], qd[:, 1:]
    w = 2.0 * (qw[:, None] * dv - dw[:, None] * qv - np.cross(qv, dv))

    _save(save_path, t, p, v, q, w)


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os, sys

    sys.path.insert(0, '/home/mpc/ros2_ws/src/QuatInnerMPC/')

    out = os.path.expanduser('~/trajectories')
    os.makedirs(out, exist_ok=True)

    generate_takeoff_trajectory(
        t_total=60.0, dt=0.02,
        p_start=[0., 0., 0.], p_target=[0., 0., 3.],
        climb_duration=4.0,
        save_path=f'{out}/takeoff.npz')

    generate_hover_trajectory(
        t_total=60.0, dt=0.02,
        p_target=[0., 0., 3.],
        save_path=f'{out}/hover.npz')

    generate_circle_trajectory(
        t_total=60.0, dt=0.02,
        radius=3.0, height=3.0, freq=0.1,
        save_path=f'{out}/circle.npz')

    generate_figure8_trajectory(
        t_total=60.0, dt=0.02,
        scale=3.0, height=3.0, freq=0.1,
        save_path=f'{out}/figure8.npz')