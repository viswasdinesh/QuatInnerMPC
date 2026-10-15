############################################################################
# run_logger.py
# Per-control-step logger for mpc_inner_quadrotor: records what the node
# sees and decides, which PX4's ULog cannot (raw MPC output in N / Nm,
# pre-clip torque, solver status and timing, input ages, predictions).
#
# One row per active control step (armed + OFFBOARD). The file is
# rewritten whenever an active window ends and on shutdown, so a crash
# loses at most the current window.
#
# Load with:  d = dict(np.load('run.npz', allow_pickle=True))
#   d['meta'].item()  → dict with run settings (weights, limits, paths)
#   field reference below (SI units, ENU / FLU unless marked FRD)
############################################################################

import os
import time

import numpy as np

FIELDS = {
    # timing (Pi wall clock, s)
    't_loop':       (),       # control callback start
    't_solved':     (),       # solve returned
    't_pub':        (),       # setpoints published
    'solve_time':   (),       # acados time_tot [s]
    'status':       (),       # acados status, 0 = success
    'traj_idx':     (),       # reference index used for stage 0
    'nav_state':    (),
    # MPC input
    'x0':           (13,),    # [p, v, q(wxyz), w]  state given to the solver
    'x_ref':        (13,),    # stage-0 reference
    # MPC output
    'u_raw':        (4,),     # [F N, τx τy τz Nm FLU]   NaN on failed solve
    'torque_unclipped': (3,), # τ_FRD / tau_max before clipping to ±1
    'thrust_norm':  (),       # published thrust (FRD z, ≤ 0)
    'torque_norm':  (3,),     # published torque (FRD, clipped)
    # input freshness: age of the newest message of each kind at t_loop [s]
    'age_att':      (),
    'age_angvel':   (),
    'age_lpos':     (),
    # PX4 sample timestamps of the messages used [µs, PX4 clock as received]
    'px4_ts_att':   (),
    'px4_ts_angvel': (),
    'px4_ts_lpos':  (),
}

PRED_FIELDS = {
    'x_pred': lambda N: (N + 1, 13),   # predicted states over the horizon
    'u_pred': lambda N: (N, 4),        # predicted inputs over the horizon
}


class RunLogger:
    def __init__(self, log_dir, N, meta=None):
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, time.strftime('%Y-%m-%d_%H-%M-%S') + '.npz')
        self.N = N
        self.meta = dict(meta or {})
        self.rows = {k: [] for k in list(FIELDS) + list(PRED_FIELDS)}
        self.window_start = []   # row index where each active window begins

    def __len__(self):
        return len(self.rows['t_loop'])

    def start_window(self):
        self.window_start.append(len(self))

    def log(self, **values):
        """Append one row. Missing fields are filled with NaN."""
        for k, shape in FIELDS.items():
            self.rows[k].append(np.asarray(values.get(k, np.full(shape, np.nan)), dtype=np.float64))
        for k, shape_fn in PRED_FIELDS.items():
            v = values.get(k)
            self.rows[k].append(np.asarray(v, dtype=np.float32) if v is not None
                                else np.full(shape_fn(self.N), np.nan, dtype=np.float32))

    def save(self):
        """Write everything logged so far (overwrites the run file). Returns path or None."""
        if len(self) == 0:
            return None
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        arrays['window_start'] = np.asarray(self.window_start, dtype=np.int64)
        tmp = self.path + '.tmp.npz'
        np.savez(tmp, meta=np.array(self.meta, dtype=object), **arrays)
        os.replace(tmp, self.path)
        return self.path
