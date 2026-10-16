############################################################################
# mpc_inner_quadrotor.py
# ROS 2 node: full-state quaternion MPC for PX4 x500 (SITL + hardware).
#
# Pipeline:
#   PX4 sensors → ENU/FLU state → Acados MPC → [F, τ] → normalise →
#   VehicleThrustSetpoint + VehicleTorqueSetpoint → PX4 control_allocator
#
# Frame conventions:
#   State / MPC:  ENU position, FLU body  (x=fwd, y=left, z=up)
#   PX4 output:   FRD body                (x=fwd, y=right, z=down)
#   Thrust:       PX4 expects normalised [-1,0] on body-z  (up = negative)
#   Torque:       PX4 expects normalised [-1,1] per axis
############################################################################

from px4_mpc.models.multirotor_inner_model import MultirotorInnerModel
from px4_mpc.controllers.multirotor_inner_mpc import MultirotorInnerMPC
from px4_mpc.utils.run_logger import RunLogger

try:
    from px4_mpc.ecrl.data.logger import ECRLLoggerMixin
except ImportError:
    # ECRL data logger not bundled with this package; logging becomes a no-op.
    class ECRLLoggerMixin:
        def ecrl_logger_init(self, *args, **kwargs):
            pass

        def ecrl_log_step(self, *args, **kwargs):
            pass

        def ecrl_save(self, *args, **kwargs):
            pass

import json
import sys
import time
from collections import deque

import rclpy
import numpy as np
import os
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

from px4_msgs.msg import (VehicleThrustSetpoint, VehicleTorqueSetpoint,
                           OffboardControlMode, VehicleStatus,
                           VehicleAttitude, VehicleLocalPosition,
                           VehicleAngularVelocity)


class MPCInnerQuadrotor(Node, ECRLLoggerMixin):

    def __init__(self):
        super().__init__('mpc_inner_quadrotor_node')

        self.ecrl_logger_init('/tmp/ecrl_log')

        # ── ROS parameters ────────────────────────────────────────────────
        self.declare_parameter('trajectory_path', '')
        traj_path = self.get_parameter('trajectory_path').value
        self.declare_parameter('log_dir', '~/mpc_logs')   # empty string disables logging
        log_dir = self.get_parameter('log_dir').value
        # Control loop rate; the MPC's first stage equals one period.
        self.declare_parameter('control_rate', 120.0)
        control_rate = float(self.get_parameter('control_rate').value)
        # Time between the state sample and the command taking effect
        # (estimator + network + allocator + motor lag). 0 disables compensation.
        self.declare_parameter('actuation_delay', 0.04)
        self.actuation_delay = float(self.get_parameter('actuation_delay').value)
        # Inputs older than one step are propagated further through the commands
        # sent since; beyond this age the node stops (heartbeat too) so PX4's
        # offboard-loss failsafe takes over.
        self.declare_parameter('max_state_age', 0.2)
        self.max_state_age = float(self.get_parameter('max_state_age').value)

        # ── QoS ───────────────────────────────────────────────────────────
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscriptions ─────────────────────────────────────────────────
        # PX4 >= v1.16 suffixes versioned messages with _v<MESSAGE_VERSION>;
        # subscribe to both names so older and newer firmware work.
        self.create_subscription(VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback, qos_sub)
        self.create_subscription(VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.status_callback, qos_sub)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback, qos_sub)
        self.create_subscription(VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.angular_velocity_callback, qos_sub)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_position_callback, qos_sub)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.local_position_callback, qos_sub)

        # ── Publishers ────────────────────────────────────────────────────
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode', qos_pub)
        self.thrust_pub = self.create_publisher(
            VehicleThrustSetpoint,
            '/fmu/in/vehicle_thrust_setpoint', qos_pub)
        self.torque_pub = self.create_publisher(
            VehicleTorqueSetpoint,
            '/fmu/in/vehicle_torque_setpoint', qos_pub)

        # ── Vehicle state  (ENU / FLU, updated by callbacks) ─────────────
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1., 0., 0., 0.])   # [qw, qx, qy, qz]
        self.w = np.zeros(3)
        self.nav_state    = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = 0  # not armed (ARMING_STATE_ARMED = VehicleStatus.ARMING_STATE_ARMED)

        # Receive time (Pi wall clock, s) and PX4 sample timestamp (µs) of the
        # newest message of each kind, for the run logger.
        self._rx_time = {'att': np.nan, 'angvel': np.nan, 'lpos': np.nan}
        self._px4_ts  = {'att': np.nan, 'angvel': np.nan, 'lpos': np.nan}

        # ── Model + MPC ───────────────────────────────────────────────────
        self.model = MultirotorInnerModel()
        self.mpc   = MultirotorInnerMPC(self.model, control_rate=control_rate)
        self.dt    = self.mpc.dt                    # control period [s]
        self.delay_steps = int(round(self.actuation_delay / self.dt))

        self.F_max   = self.model.max_thrust   # 34.194 N
        self.tau_max = self.model.tau_max      # [2.975, 2.975, 0.547] Nm

        # ── Trajectory ────────────────────────────────────────────────────
        self._load_trajectory(traj_path)

        self.traj_idx = 0
        self._solver_warmed_up = False  # set False whenever guard blocks; triggers warm-start on re-entry

        # ── Delay compensation ────────────────────────────────────────────
        # Commands sent but not yet acting on the measured state, oldest first.
        self.u_hover = np.array([self.model.mass * 9.81, 0.0, 0.0, 0.0])
        self._max_stale_steps = int(np.ceil(self.max_state_age / self.dt))
        self._u_hist = deque(maxlen=max(self.delay_steps + self._max_stale_steps, 1))
        self._last_stale_warn = 0.0
        self.mpc.integrator.set('p', np.zeros(self.model.get_acados_model().p.size()[0]))

        # ── Run logger ────────────────────────────────────────────────────
        self._active = False   # inside an armed + OFFBOARD window
        self.run_log = RunLogger(log_dir, self.mpc.N, self._log_meta(traj_path)) if log_dir else None
        if self.run_log is not None:
            self.get_logger().info(f'Logging control steps to {self.run_log.path}')

        # ── Control timer: control_rate (= first MPC stage) ─────────────────
        self.create_timer(self.dt, self.cmdloop_callback)

        self.get_logger().info(
            f'MPCInnerQuadrotor ready at {1.0 / self.dt:g} Hz '
            f'(stages {sorted(set(np.round(self.mpc.time_steps * 1e3, 2)))} ms, horizon {self.mpc.Tf:.2f} s). '
            f'Trajectory length: {self.traj_len} steps '
            f'({self.traj_len * self.dt:.1f} s), delay compensation: {self.delay_steps} step(s) '
            f'({self.delay_steps * self.dt * 1e3:.0f} ms), max state age {self.max_state_age * 1e3:.0f} ms')

    # ──────────────────────────────────────────────────────────────────────
    #  Trajectory loading
    # ──────────────────────────────────────────────────────────────────────

    def _load_trajectory(self, path: str):
        """
        Load a pre-computed trajectory from a .npz file ('~' is expanded).
        An empty path selects a static hover at z=3 m; a path that does not
        exist is an error, so a typo cannot silently turn into a hover.

        Expected keys:  p (M,3), v (M,3), q (M,4), w (M,3)
        Optional key:   t (M,)  sample times; default 0.02 s spacing. The file
                        is resampled to the control period if it differs.
        """
        path = os.path.expanduser(path) if path else ''
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f'trajectory_path does not exist: {path}')
        if path:
            traj = np.load(path)
            self.traj_p = traj['p'].astype(np.float64)   # (M, 3)
            self.traj_v = traj['v'].astype(np.float64)   # (M, 3)
            self.traj_q = traj['q'].astype(np.float64)   # (M, 4)
            self.traj_w = traj['w'].astype(np.float64)   # (M, 3)
            t_src = traj['t'].astype(np.float64) if 't' in traj else 0.02 * np.arange(len(self.traj_p))
            self._resample_trajectory(t_src)
            self.get_logger().info(
                f'Loaded trajectory from {path} ({len(t_src)} samples at '
                f'{(t_src[1] - t_src[0]) * 1e3:.2f} ms → {self.traj_p.shape[0]} at {self.dt * 1e3:.2f} ms)')
        else:
            # Default: static hover at z = 3 m
            M = int(round(60.0 / self.dt))   # 60 s — more than enough for any test
            self.traj_p = np.tile([0., 0., 3.], (M, 1))
            self.traj_v = np.zeros((M, 3))
            self.traj_q = np.tile([1., 0., 0., 0.], (M, 1))
            self.traj_w = np.zeros((M, 3))
            self.get_logger().warn(
                'No trajectory_path given — using static hover at z=3 m.')

        self.traj_len = self.traj_p.shape[0]

    def _resample_trajectory(self, t_src):
        """Resample the loaded trajectory onto the control period (index = step)."""
        t_src = t_src - t_src[0]
        if len(t_src) > 1 and abs((t_src[1] - t_src[0]) - self.dt) < 1e-9:
            return
        t_new = np.arange(0.0, t_src[-1] + 1e-9, self.dt)
        lin = lambda a: np.stack([np.interp(t_new, t_src, a[:, i]) for i in range(a.shape[1])], axis=1)
        q = self.traj_q.copy()
        for i in range(1, len(q)):                       # keep one hemisphere before interpolating
            if np.dot(q[i], q[i - 1]) < 0.0:
                q[i] = -q[i]
        q = lin(q)
        self.traj_q = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.traj_p, self.traj_v, self.traj_w = lin(self.traj_p), lin(self.traj_v), lin(self.traj_w)

    def _log_meta(self, traj_path):
        meta = {
            'trajectory_path': traj_path, 'dt': self.dt, 'N': self.mpc.N, 'Tf': self.mpc.Tf,
            'mass': self.model.mass, 'kf': self.model.kf,
            'omega_min': self.model.omega_min, 'omega_max': self.model.omega_max,
            'F_max': self.F_max, 'tau_max': np.asarray(self.tau_max).tolist(),
            'delay_steps': self.delay_steps, 'actuation_delay': self.actuation_delay,
            'control_rate': 1.0 / self.dt, 'time_steps': self.mpc.time_steps.tolist(),
            'stage_offsets': self.mpc.stage_offsets.tolist(),
            'max_state_age': self.max_state_age,
        }
        # Cost weights as compiled into the loaded solver.
        mpc_dir = os.path.dirname(sys.modules[MultirotorInnerMPC.__module__].__file__)
        json_path = os.path.join(mpc_dir, '..', 'mpc_codegen', 'acados_ocp.json')
        try:
            with open(json_path) as f:
                cost = json.load(f)['cost']
            meta['W_diag'] = np.diag(cost['W']).tolist()
            meta['W_e_diag'] = np.diag(cost['W_e']).tolist()
        except (OSError, KeyError, ValueError):
            pass
        return meta

    # ──────────────────────────────────────────────────────────────────────
    #  Subscriber callbacks
    # ──────────────────────────────────────────────────────────────────────

    def status_callback(self, msg):
        self.nav_state    = msg.nav_state
        self.arming_state = msg.arming_state

    def _stamp(self, key, msg):
        self._rx_time[key] = time.time()
        self._px4_ts[key] = msg.timestamp_sample

    def attitude_callback(self, msg):
        self._stamp('att', msg)

        qw, qx, qy, qz = msg.q
        q_enu = (1.0 / np.sqrt(2)) * np.array([
            qw + qz,
            qx + qy,
            qx - qy,
            qw - qz,
        ])
        self.q = q_enu / np.linalg.norm(q_enu)

    def angular_velocity_callback(self, msg):
        """
        PX4 publishes body rates in FRD.
        FRD → FLU: flip y and z axes.
        """
        self._stamp('angvel', msg)
        wx, wy, wz = msg.xyz
        self.w = np.array([wx, -wy, -wz])

    def local_position_callback(self, msg):
        """
        PX4 publishes position/velocity in NED.
        NED → ENU: (x,y,z)_NED → (y, x, −z)_ENU
        """
        self._stamp('lpos', msg)
        self.p = np.array([ msg.y,  msg.x, -msg.z ])
        self.v = np.array([ msg.vy, msg.vx, -msg.vz])

    # ──────────────────────────────────────────────────────────────────────
    #  Main control loop  (control_rate, default 120 Hz)
    # ──────────────────────────────────────────────────────────────────────

    def cmdloop_callback(self):
        t_loop = time.time()

        # Age of the stalest state input (inf before the first message).
        ages = [t_loop - t for t in self._rx_time.values()]
        state_age = np.inf if np.isnan(ages).any() else max(ages)
        too_stale = state_age > self.max_state_age

        # ── 1. Offboard heartbeat ──────────────────────────────────────────
        # Withheld while flying on a too-old state, so PX4's offboard-loss
        # failsafe (COM_OF_LOSS_T, COM_OBL_RC_ACT) takes over.
        # direct_actuator = True routes through control_allocator,
        # bypassing PX4's attitude and rate controllers entirely.
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp       = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position        = False
        offboard_msg.velocity        = False
        offboard_msg.acceleration    = False
        offboard_msg.attitude        = False
        offboard_msg.body_rate       = False
        offboard_msg.direct_actuator = False
        offboard_msg.thrust_and_torque = True
        if not (too_stale and self._active):
            self.offboard_mode_pub.publish(offboard_msg)

        # ── Guard: skip solve until armed and in offboard mode ─────────────
        # The offboard heartbeat above must always publish (PX4 requires it
        # continuously to allow/maintain OFFBOARD mode entry). But we must
        # not command thrust before the motors are live — doing so causes the
        # MPC to warm-start toward a saturated solution that immediately
        # flips the vehicle on arming.
        if (self.nav_state    != VehicleStatus.NAVIGATION_STATE_OFFBOARD or
                self.arming_state != VehicleStatus.ARMING_STATE_ARMED):
            self._solver_warmed_up = False
            if self._active:
                self._active = False
                self._save_run_log()
            return

        if not self._active:
            self._active = True
            # Before OFFBOARD, PX4's own controllers were holding a hover.
            self._u_hist.clear()
            self._u_hist.extend([self.u_hover] * self._u_hist.maxlen)
            if self.run_log is not None:
                self.run_log.start_window()

        # ── 2. State vector  [p, v, q, w]  shape (13,) ────────────────────
        x0 = np.concatenate([self.p, self.v, self.q, self.w])

        if too_stale:
            # No usable state: publish nothing (PX4 holds the last command until
            # its offboard-loss timeout) and re-seed the solver on recovery.
            self._solver_warmed_up = False
            self._u_hist.append(self._u_hist[-1])
            if t_loop - self._last_stale_warn > 1.0:
                self._last_stale_warn = t_loop
                self.get_logger().warn(
                    f'State {state_age * 1e3:.0f} ms old (> {self.max_state_age * 1e3:.0f} ms): '
                    'not commanding, heartbeat withheld.')
            if self.run_log is not None:
                self.run_log.log(t_loop=t_loop, traj_idx=self.traj_idx, nav_state=self.nav_state,
                                 x0=x0, status=-2, stale_steps=np.nan,
                                 **self._age_fields(t_loop))
            self.traj_idx = min(self.traj_idx + 1, self.traj_len - 1)
            return

        # ── Delay compensation: plan from where the vehicle will be when this
        # command takes effect, by replaying the commands still in flight.
        # A stale measurement (no new message for > 1 step) is propagated
        # further, through the commands sent since it arrived.
        stale_steps = int(max(state_age - self.dt, 0.0) // self.dt)
        x_comp = self._propagate(x0, self.delay_steps + stale_steps)

        # ── Warm-start on first entry into active control ──────────────────
        # Acados initialises all stages to the x0 from setup (z=0, level).
        # If the vehicle is already in flight (e.g. handed off from PX4
        # position controller), that cold trajectory is a poor linearisation
        # point and causes the first QP to fail → solver returns status≠0 →
        # exception → no commands published → PX4 failsafe.
        # Fix: seed every stage with the actual current state + hover thrust.
        if not self._solver_warmed_up:
            F_hover = self.model.mass * 9.81   # ~19.6 N
            u_init  = np.array([F_hover, 0.0, 0.0, 0.0])
            for i in range(self.mpc.N + 1):
                self.mpc.ocp_solver.set(i, "x", x_comp)
            for i in range(self.mpc.N):
                self.mpc.ocp_solver.set(i, "u", u_init)
            self._solver_warmed_up = True
            self.get_logger().info('MPC warm-started with current state.')

        # ── 3. Per-stage reference parameters ─────────────────────────────
        # Each stage k receives the reference at time t + k·dt so the MPC
        # looks ahead along the trajectory, not just at a static point.
        # model.p = [p_ref(3), v_ref(3), q_ref(4), w_ref(3)]  → 13 values
        # Stage 0 is delay_steps ahead of the measurement, so is its reference;
        # stage k starts stage_offsets[k] control periods after stage 0.
        for k in range(self.mpc.N + 1):
            idx = min(self.traj_idx + self.delay_steps + self.mpc.stage_offsets[k], self.traj_len - 1)
            p_param = np.concatenate([
                self.traj_p[idx],   # p_ref
                self.traj_v[idx],   # v_ref
                self.traj_q[idx],   # q_ref
                self.traj_w[idx],   # w_ref
            ])
            self.mpc.ocp_solver.set(k, "p", p_param)

        # ── 4. Solve ───────────────────────────────────────────────────────
        idx0 = min(self.traj_idx + self.delay_steps, self.traj_len - 1)
        log_row = dict(
            t_loop=t_loop, traj_idx=self.traj_idx, nav_state=self.nav_state, x0=x0, x_comp=x_comp,
            x_ref=np.concatenate([self.traj_p[idx0], self.traj_v[idx0],
                                  self.traj_q[idx0], self.traj_w[idx0]]),
            stale_steps=stale_steps, **self._age_fields(t_loop),
        )
        try:
            simU, simX = self.mpc.solve(x_comp)
        except Exception as e:
            self.get_logger().warn(f'MPC solve failed: {e}')
            self._solver_warmed_up = False   # force re-seed on next call
            # Nothing is published, so PX4 keeps applying the previous command.
            self._u_hist.append(self._u_hist[-1])
            if self.run_log is not None:
                self.run_log.log(**log_row, t_solved=time.time(),
                                 status=self.mpc.ocp_solver.get_status(),
                                 solve_time=self.mpc.ocp_solver.get_stats('time_tot'))
            return
        t_solved = time.time()

        # ── 5. Extract first optimal control  u0 = [F, τ_x, τ_y, τ_z] ────
        u0    = simU[0, :]
        F_cmd = float(u0[0])          # [N]   total thrust
        tau   = u0[1:4].astype(float) # [Nm]  body torques in FLU

        # ── 6. Normalise thrust → [−1, 0]  (PX4 FRD body-z convention) ───
        # PX4: thrust_body[2] < 0 means upward thrust in FRD (z points down)
        # With THR_MDL_FAC=0 PX4 passes the setpoint straight to the motors,
        # and gz maps it to ω = EC_MIN + (EC_MAX − EC_MIN)·u with F = kf·ω²,
        # so invert that curve instead of assuming F ∝ u.
        omega_req   = np.sqrt(max(F_cmd, 0.0) / (4.0 * self.model.kf))
        u_thrust    = (omega_req - self.model.omega_min) / (self.model.omega_max - self.model.omega_min)
        thrust_norm = float(-np.clip(u_thrust, 0.0, 1.0))

        # ── 7. Normalise torques → [−1, 1], convert FLU → FRD ─────────────
        # FLU → FRD: y and z axes flip sign
        tau_frd = np.array([tau[0], -tau[1], -tau[2]])
        torque_unclipped = tau_frd / self.tau_max
        torque_norm = np.clip(torque_unclipped, -1.0, 1.0)

        # ── 8. Publish ─────────────────────────────────────────────────────
        ts = int(Clock().now().nanoseconds / 1000)

        thrust_msg         = VehicleThrustSetpoint()
        thrust_msg.timestamp = ts
        thrust_msg.xyz     = [0.0, 0.0, thrust_norm]
        self.thrust_pub.publish(thrust_msg)

        torque_msg         = VehicleTorqueSetpoint()
        torque_msg.timestamp = ts
        torque_msg.xyz     = torque_norm.tolist()
        self.torque_pub.publish(torque_msg)

        if self.run_log is not None:
            self.run_log.log(**log_row, t_solved=t_solved, t_pub=time.time(), status=0,
                             solve_time=self.mpc.ocp_solver.get_stats('time_tot'),
                             u_raw=u0, torque_unclipped=torque_unclipped,
                             thrust_norm=thrust_norm, torque_norm=torque_norm,
                             x_pred=simX, u_pred=simU)

        # ── 9. ECRL data logging ───────────────────────────────────────────
        # State for ECRL: [p, v, q] (drop ω — rate-model convention).
        # Control for ECRL: [aT, ω_measured] where aT = F/mass and ω is
        # the measured body rate (best proxy for effective angular rate command
        # at 50 Hz when the inner loop tracks well).
        idx_d = min(self.traj_idx, self.traj_len - 1)
        x_ecrl   = np.concatenate([self.p, self.v, self.q])          # R10
        x_d_ecrl = np.concatenate([self.traj_p[idx_d],
                                    self.traj_v[idx_d],
                                    self.traj_q[idx_d]])              # R10
        u_ecrl   = np.array([F_cmd / self.model.mass,
                              self.w[0], self.w[1], self.w[2]])       # R4 [aT, ω_flu]
        u_d_ecrl = np.array([9.81,
                              self.traj_w[idx_d][0],
                              self.traj_w[idx_d][1],
                              self.traj_w[idx_d][2]])                 # R4 [g, ω_ref]
        self.ecrl_log_step(x_ecrl, x_d_ecrl, u_ecrl, u_d_ecrl)

        self._u_hist.append(u0.astype(float))

        # ── 10. Advance trajectory index ───────────────────────────────────
        # Advance every control step; hold at last point when done.
        self.traj_idx = min(self.traj_idx + 1, self.traj_len - 1)

    def _age_fields(self, t_loop):
        return dict(age_att=t_loop - self._rx_time['att'],
                    age_angvel=t_loop - self._rx_time['angvel'],
                    age_lpos=t_loop - self._rx_time['lpos'],
                    px4_ts_att=self._px4_ts['att'],
                    px4_ts_angvel=self._px4_ts['angvel'],
                    px4_ts_lpos=self._px4_ts['lpos'])

    def _propagate(self, x, n):
        """Integrate x forward through the last n commands sent (one dt each)."""
        if n <= 0:
            return x
        x = x.copy()
        for u in list(self._u_hist)[-n:]:
            x = self.mpc.integrator.simulate(x=x, u=u)
            x[6:10] /= np.linalg.norm(x[6:10])
        return x

    def _save_run_log(self):
        if self.run_log is not None:
            path = self.run_log.save()
            if path:
                self.get_logger().info(f'Saved {len(self.run_log)} control steps to {path}')


# ──────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MPCInnerQuadrotor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._save_run_log()
        node.ecrl_save()          # flush collected samples to /tmp/ecrl_log/
        try:                      # launch may send a second SIGINT while tearing down
            node.destroy_node()
            rclpy.try_shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass


if __name__ == '__main__':
    main()