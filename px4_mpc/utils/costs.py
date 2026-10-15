import casadi as cs

from . import quaternion

def quat_error_cost(qd, q, Qq):
    dq = quaternion.quat_error(qd, q)
    dqv = dq[1:]  
    return cs.mtimes(dqv.T, cs.mtimes(Qq, dqv))

def position_cost(pd, p, Qp):
    e_p = pd - p
    return cs.mtimes(e_p.T, cs.mtimes(Qp, e_p))

def velocity_cost(vd, v, Qv):
    e_v = vd - v
    return cs.mtimes(e_v.T, cs.mtimes(Qv, e_v))

def rate_cost(wd, w, Qw):
    e_w = wd - w
    return cs.mtimes(e_w.T, cs.mtimes(Qw, e_w))

def control_cost(u, R):
    return cs.mtimes(u.T, cs.mtimes(R, u))