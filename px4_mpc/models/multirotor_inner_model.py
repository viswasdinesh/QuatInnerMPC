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
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

from acados_template import AcadosModel
import casadi as cs
import numpy as np

class MultirotorInnerModel():
    def __init__(self):
        self.name = 'multirotor_inner_model'

        # constants
        self.mass = 2.0
        self.J = cs.diag(cs.vertcat(0.02166666666666667, 0.02166666666666667, 0.04))
        self.j_inv = cs.inv(self.J)
        hover_thrust = 0.73
        # self.max_thrust = self.mass * 9.81/hover_thrust
        # self.max_rate = 0.5

        # ── From x500/model.sdf (motor plugin) ─────────────────────────
        self.kf        = 8.54858e-06           # N/(rad/s)²  motorConstant
        self.km        = 0.016                 # dimensionless  momentConstant
        self.kt        = self.km * self.kf     # Nm/(rad/s)²  = 1.3677728e-07
        self.omega_max = 1000.0                # rad/s  maxRotVelocity (SIM_GZ_EC_MAX)
        self.omega_min = 150.0                 # rad/s  SIM_GZ_EC_MIN (idle speed when armed)

        self.max_thrust = 4 * self.kf * self.omega_max**2  # 34.194 N  (4 motors)
        self.max_rate   = 3.14                  # rad/s  (keep as soft limit for MPC)

        # ── Allocation matrix (FRD body, u = B @ f, f in [N]) ──────────
        # Rotor positions from x500_base:
        #   rotor_0: ( 0.174, -0.174)  CCW  spin=+1
        #   rotor_1: (-0.174,  0.174)  CCW  spin=+1
        #   rotor_2: ( 0.174,  0.174)  CW   spin=-1
        #   rotor_3: (-0.174, -0.174)  CW   spin=-1
        #
        # τ_x (roll)  =  Σ fi * (−y_i)
        # τ_y (pitch) =  Σ fi * ( x_i)
        # τ_z (yaw)   =  Σ spin_i * km * fi

        self.B = np.array([
            [ 1.0,    1.0,    1.0,    1.0   ],  # thrust
            [ 0.174, -0.174, -0.174,  0.174 ],  # roll
            [ -0.174, 0.174,  -0.174, 0.174 ],  # pitch
            [ 0.016,  0.016, -0.016, -0.016 ],  # yaw (km * fi)
        ])
        self.B_inv = np.linalg.pinv(self.B)

        # Max torques for normalisation (differential max, 2 motors)
        F_m = self.kf * self.omega_max**2      # 8.5486 N per motor
        self.tau_max = np.array([
            0.174 * F_m * 2,                   # roll  2.974 Nm
            0.174 * F_m * 2,                   # pitch 2.974 Nm
            self.km * F_m * 4,                 # yaw   0.547 Nm
        ])

    def get_acados_model(self) -> AcadosModel:
        def skew_symmetric(v):
            return cs.vertcat(cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0))

        def q_to_rot_mat(q):
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]

            rot_mat = cs.vertcat(
                cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
                cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

            return rot_mat

        def v_dot_q(v, q):
            rot_mat = q_to_rot_mat(q)

            return cs.mtimes(rot_mat, v)

        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3)
        v      = cs.MX.sym('v', 3)
        q      = cs.MX.sym('q', 4)
        w      = cs.MX.sym('w', 3)

        x = cs.vertcat(p, v, q, w)

        F = cs.MX.sym('F')
        tau_x = cs.MX.sym('tau_x')
        tau_y = cs.MX.sym('tau_y')
        tau_z = cs.MX.sym('tau_z')
        
        tau = cs.vertcat(tau_x, tau_y, tau_z)

        u = cs.vertcat(F, tau)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)
        w_dot      = cs.MX.sym('w_dot', 3)

        xdot = cs.vertcat(p_dot, v_dot, q_dot, w_dot)
        g = cs.vertcat(0.0, 0.0, -9.81) # gravity constant [m/s^2]
        
        force = cs.vertcat(0.0, 0.0, F)

        a_thrust = v_dot_q(force, q)/self.mass

        q_dot = 0.5* cs.mtimes(skew_symmetric(w), q)

        Jw = cs.mtimes(self.J, w)

        w_dot = cs.mtimes(self.j_inv, tau - cs.cross(w, Jw))
        
        # dynamics
        f_expl = cs.vertcat(v,
                        a_thrust + g,
                        q_dot,
                        w_dot
                        )

        f_impl = xdot - f_expl

        # Set params
        
        p_ref = cs.MX.sym('p_ref', 3)
        v_ref = cs.MX.sym('v_ref', 3)
        q_ref = cs.MX.sym('q_ref', 4)
        w_ref = cs.MX.sym('w_ref', 3)

        model.p = cs.vertcat(p_ref, v_ref, q_ref, w_ref)

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model
