import casadi as cs

def quat_conjugate(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return cs.vertcat(qw, -qx, -qy, -qz)

def quat_norm(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return cs.sqrt(qw**2 + qx**2 + qy**2 + qz**2)

def quat_inverse(q):
    return quat_conjugate(q) / quat_norm(q) ** 2

def quat_mul(q1, q2):
    qw1 = q1[0]
    qw2 = q2[0]

    vec1 = cs.vertcat(q1[1], q1[2], q1[3])
    vec2 = cs.vertcat(q2[1], q2[2], q2[3])

    qw = qw1 * qw2 - cs.dot(vec1, vec2)
    vec = qw1 * vec2 + qw2 * vec1 + cs.cross(vec1, vec2)
    return cs.vertcat(qw, vec)

def quat_antipodal_correct(qd, q):
    """
    Resolve the quaternion double-cover ambiguity: qd and -qd represent the
    same physical orientation, but as 4-vectors they are far apart. Pick the
    sign of qd that lies in the same hemisphere as q (dot(qd, q) >= 0) so the
    error quaternion always represents the short-way rotation, instead of
    occasionally being pulled toward the antipodal (180 deg the "long way")
    solution.
    """
    return cs.if_else(cs.dot(qd, q) < 0, -qd, qd)

def quat_error(qd, q):
    # Normalise q so the error depends only on attitude, not on |q|; otherwise
    # the cost scales with |q|^2 and rewards shrinking the quaternion.
    # (qd is already normalised by quat_inverse.)
    q = q / quat_norm(q)
    qd = quat_antipodal_correct(qd, q)
    return quat_mul(quat_inverse(qd), q)
    
def unit_norm_error(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    norm_error = qw**2 + qx**2 + qy**2 + qz**2 - 1.0
    return norm_error

