############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, E# VEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import scipy.linalg
import casadi as cs
import os

from px4_mpc.utils.costs import quat_error_cost, position_cost, velocity_cost, rate_cost, control_cost
from px4_mpc.utils import quaternion

def time_grid(control_rate, horizon=0.4, N=20, min_fine=4):
    """Stage lengths for an N-stage horizon run at control_rate.

    The first stages equal the control period (so the integrator and the
    feedback act on one period); the remaining stages are an integer multiple
    L of it, so every stage starts a whole number of control periods ahead.
    Returns (time_steps, offsets) with offsets[k] = start of stage k in periods.
    """
    dt = 1.0 / control_rate
    M = int(round(horizon / dt))            # horizon in control periods
    if M <= N:                              # uniform grid (e.g. 50 Hz: 20 × 20 ms)
        steps = np.ones(N, dtype=int)
    else:
        for L in range(2, 11):
            n_long, rem = divmod(M - N, L - 1)
            if rem == 0 and N - n_long >= min_fine:
                steps = np.r_[np.ones(N - n_long, dtype=int), np.full(n_long, L)]
                break
        else:
            raise ValueError(f'No {N}-stage grid of whole control periods spans '
                             f'{horizon} s at {control_rate} Hz')
    return steps * dt, np.r_[0, np.cumsum(steps)]


class MultirotorInnerMPC():
    def __init__(self, model, control_rate=120.0):
        self.model = model
        self.N = 20
        self.dt = 1.0 / control_rate                    # control period = first stage
        self.time_steps, self.stage_offsets = time_grid(control_rate, horizon=0.4, N=self.N)
        self.Tf = float(np.sum(self.time_steps))

        self.x0 = np.array([0.0, 0.0, 0.0,          # p
                            0.0, 0.0, 0.0,          # v
                            1.0, 0.0, 0.0, 0.0,     # q
                            0.0, 0.0, 0.0])         # w

        self.ocp_solver, self.integrator = self.setup(self.x0, self.N , self.Tf)

    def setup(self, x0, N_horizon, Tf):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Set directory for code generation and json file
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        codegen_dir = os.path.join(package_root, 'mpc_codegen')
        json_path = os.path.join(codegen_dir, 'acados_ocp.json')
        os.makedirs(codegen_dir, exist_ok=True)
        ocp.code_export_directory = codegen_dir

        # set model
        model = self.model.get_acados_model()
        Fmax = self.model.max_thrust
        wmax = self.model.max_rate

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        # Cached for solve(): get_acados_model() rebuilds the CasADi model
        # (~1 ms per call), so it must not be called in the control loop.
        self.nx, self.nu = nx, nu
        ny = nx + nu
        ny_e = nx

        # set dimensions
        ocp.solver_options.N_horizon = N_horizon

        # set cost
        Q_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 0.0, 0.1, 0.1, 0.1])
        Q_e = 2*np.diag([3e2, 3e2, 3e2, 1e2, 1e2, 1e2, 0.0, 0.0, 0.0, 0.0])
        R_mat = 2*np.diag([1e1, 5e2, 5e2, 5e2])
        Q_quat = np.diag([10.0, 10.0, 10.0, 10.0])

        # TODO: How do you add terminal costs?

        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
        # ocp.cost.cost_type = 'EXTERNAL'
        # ocp.cost.cost_type_e = 'EXTERNAL'
        # ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
        # ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_e @ model.x

        # ocp.cost.cost_type = 'NONLINEAR_LS'
        # ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        # ocp.cost.W_e = scipy.linalg.block_diag(Q_e)




        # ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        # ocp.model.cost_y_expr_e = model.x
        # ocp.cost.yref  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # set weights
        Q_p = np.diag([100.0, 100.0, 100.0])
        Q_v = np.diag([20.0, 20.0, 20.0])
        Q_q = np.diag([10.0, 10.0, 10.0])
        Q_w = np.diag([1, 10, 5])
        R   = np.diag([10, 50, 100, 50])

        # Penalise deviation from hover, not absolute thrust — otherwise the
        # optimiser trades altitude for a cheaper (lower) thrust and sags.
        u_hover = np.array([self.model.mass * 9.81, 0.0, 0.0, 0.0])

        # ── Exact-Hessian formulation (EXTERNAL cost) ─────────────────────
        # Disabled: ~93% of solve time went to second-order sensitivities.
        # Re-enable together with hessian_approx = 'EXACT' and
        # regularize_method = 'CONVEXIFY' below.
        # Jp = position_cost(model.p[0:3], model.x[0:3], Q_p)
        # Jv = velocity_cost(model.p[3:6], model.x[3:6], Q_v)
        # Jq = quat_error_cost(model.p[6:10], model.x[6:10], Q_q)
        # Jw = rate_cost(model.p[10:13], model.x[10:13], Q_w)
        # Ju = control_cost(model.u - u_hover, R)
        #
        # stage_cost = Jp + Jv + Jq + Jw + Ju
        # terminal_cost = Jp + Jv + Jq + Jw
        #
        # ocp.cost.cost_type = 'EXTERNAL'
        # ocp.cost.cost_type_e = 'EXTERNAL'    # Terminal cost
        # ocp.model.cost_expr_ext_cost = stage_cost
        # ocp.model.cost_expr_ext_cost_e = terminal_cost

        # ── Gauss-Newton formulation (NONLINEAR_LS cost) ──────────────────
        # Residuals are built from the per-stage reference parameters, so
        # yref = 0. acados minimises ½‖y − yref‖²_W, so W = 2·Q gives the
        # same objective as the EXTERNAL cost above.
        e_p = model.x[0:3] - model.p[0:3]
        e_v = model.x[3:6] - model.p[3:6]
        e_q = quaternion.quat_error(model.p[6:10], model.x[6:10])[1:]
        e_w = model.x[10:13] - model.p[10:13]
        e_u = model.u - u_hover

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = cs.vertcat(e_p, e_v, e_q, e_w, e_u)
        ocp.model.cost_y_expr_e = cs.vertcat(e_p, e_v, e_q, e_w)
        ocp.cost.W = 2 * scipy.linalg.block_diag(Q_p, Q_v, Q_q, Q_w, R)
        ocp.cost.W_e = 2 * scipy.linalg.block_diag(Q_p, Q_v, Q_q, Q_w)
        ocp.cost.yref = np.zeros(16)
        ocp.cost.yref_e = np.zeros(12)



        # set constraints
        ocp.constraints.lbu = np.array([0.0, -self.model.tau_max[0], -self.model.tau_max[1], -self.model.tau_max[2]])
        ocp.constraints.ubu = np.array([+Fmax,  self.model.tau_max[0], self.model.tau_max[1], self.model.tau_max[2]])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ocp.constraints.x0 = x0

        ocp.parameter_values = np.zeros(model.p.size()[0])

        # set options
        # ocp.solver_options.regularize_method = 'CONVEXIFY'  # exact Hessian only
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        # ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 1
        use_RTI=True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
        else:
            ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

        ocp.solver_options.qp_solver_cond_N = N_horizon

        # set prediction horizon
        ocp.solver_options.tf = Tf
        ocp.solver_options.time_steps = self.time_steps

        # Reuse the compiled solvers if present. acados hashes the OCP and rebuilds it
        # when anything changed; it does not check an integrator created from an
        # AcadosOcp, so rebuild that whenever the OCP was rebuilt (same model, tf, N).
        prebuilt = all(os.path.isfile(os.path.join(codegen_dir, f'libacados_{s}_solver_{model.name}.so'))
                       for s in ('ocp', 'sim'))
        ocp_solver = AcadosOcpSolver(ocp, json_file=json_path, generate=not prebuilt, build=not prebuilt)
        rebuild_sim = not prebuilt or ocp_solver.generated
        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file=json_path, generate=rebuild_sim, build=rebuild_sim)

        return ocp_solver, acados_integrator
    
    def solve(self, x0, verbose=False):

        # preparation phase
        ocp_solver = self.ocp_solver

        # set initial state
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        status = ocp_solver.solve()
        if verbose:
            self.ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        N = self.N
        nx, nu = self.nx, self.nu

        simX = np.ndarray((N+1, nx))
        simU = np.ndarray((N, nu))

        # get solution
        for i in range(N):
            simX[i,:] = self.ocp_solver.get(i, "x")
            simU[i,:] = self.ocp_solver.get(i, "u")
        simX[N,:] = self.ocp_solver.get(N, "x")

        return simU, simX
    
