"""
Examples
--------
>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>> qx = quaternion_from_axis_angle(axis=xaxis, angle=alpha)
>>> qy = quaternion_from_axis_angle(axis=yaxis, angle=beta)
>>> qz = quaternion_from_axis_angle(axis=zaxis, angle=gamma)
>>> q = quaternion_multiply(qx, qy)
>>> q = quaternion_multiply(q, qz)
>>> Rq = quaternion_to_transform_matrix(q)
>>> is_same_transform(R, Rq)
True

"""
from math import cos, sin, sqrt, acos, atan2, tau, asin, pi

import numpy as np

from lib.numba_opt import double, jit_hardcore
from lib.stuff import EPS
from lib.transformations.euler_angles import euler_from_matrix, wrap_angle
from lib.transformations.transform_tools import _NEXT_AXIS, AXES2TUPLE, is_same_transform, transform_matrix
from lib.vectors import vector_normalize, orthogonal, norm, origin, xaxis, yaxis, zaxis

_default_q = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)

quaternion = np.ndarray


@jit_hardcore
def unit() -> quaternion:
    return _default_q.copy()


@jit_hardcore
def euler_from_quaternion(q: quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion(quaternion_from_elements(0.99810947, 0.06146124, 0, 0))
    >>> np.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_to_transform_matrix(q), axes)


@jit_hardcore
def quaternion_from_euler(ai: float, aj: float, ak: float, axes: str = 'sxyz') -> quaternion:
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> _q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(_q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    firstaxis, parity, repetition, frame = AXES2TUPLE(axes)

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = cos(ai)
    si = sin(ai)
    cj = cos(aj)
    sj = sin(aj)
    ck = cos(ak)
    sk = sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,), dtype=double)
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    return q


@jit_hardcore
def quaternion_from_elements(w: float, i: float, j: float, k: float) -> quaternion:
    """
    Build quaternion from components
    :param w:
    :param i:
    :param j:
    :param k:
    :return: quaternion as np array
    """
    return np.array((w, i, j, k), dtype=double)


@jit_hardcore
def quaternion_to_transform_matrix(q: quaternion) -> transform_matrix:
    """Return homogeneous rotation matrix from quaternion. Quaternion must be in a form of np array
    >>> M = quaternion_to_transform_matrix(quaternion_from_elements(1, 0, 0, 0))
    >>> np.allclose(M, np.ident(4))
    True
    >>> M = quaternion_to_transform_matrix(quaternion_from_elements(0, 1, 0, 0))
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """

    q = np.copy(q)
    n = np.dot(q, q)
    if n < EPS:
        return np.identity(4)
    q *= sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


@jit_hardcore
def _trace_method(matrix: transform_matrix) -> quaternion:
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    """
    m = matrix.conj().transpose()  # This method assumes row-vector and postmultiplication of that vector
    q = np.zeros(4, dtype=double)
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q[:] = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q[:] = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q[:] = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q[:] = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

    q *= 0.5 / sqrt(t)
    return q


@jit_hardcore
def quaternion_from_matrix(matrix: np.ndarray, atol: float = 1e-08, skip_checks=False):
    """Return quaternion from rotation/transformation matrix.

    Create q vector by specifying the 3x3 rotation or 4x4 transformation matrix
        (as a np array) from which the quaternion's rotation should be created.


    >>> q = quaternion_from_matrix(np.ident(4))
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1.0, -1.0, -1.0, 1.0]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True

    """

    shape = matrix.shape

    if shape == (3, 3):
        R = matrix
    elif shape == (4, 4):
        R = matrix[:-1][:, :-1]  # Upper left 3x3 sub-matrix
    else:
        raise ValueError("Invalid matrix shape: Input must be a 3x3 or 4x4 np array or matrix")
    if not skip_checks:
        # Check matrix properties
        if np.any(np.abs(np.dot(R, R.conj().transpose()) - np.eye(3)) > atol):
            raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse")
        if np.abs(np.linalg.det(R) - 1.0 > atol):
            raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0")

    return _trace_method(R)


@jit_hardcore
def quaternion_to_transformation_matrix(q: np.ndarray):
    """Get the 4x4 homogeneous transformation matrix equivalent of the quaternion rotation.

    Returns:
        A 4x4 homogeneous transformation matrix as a 4x4 Numpy array

    Note:
        This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly
         normalise the Quaternion object to a unit quaternion if it is not already one.
    """
    q = np.copy(q)
    n = np.dot(q, q)
    if n < EPS:
        return np.identity(4)
    q *= sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def quaternion_to_str(q):
    """An informal, nicely printable string representation of the Quaternion object.
   """
    return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(*q)


@jit_hardcore
def quaternion_inverse(q):
    """Inverse of the quaternion object, encapsulated in a new instance.

    For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result
     in the null rotation.

    Returns:
        A new Quaternion object representing the inverse of this object

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> np.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True
    """
    ss = np.dot(q, q)
    if ss > 0:
        return quaterion_vector_conjugate(q) / ss
    else:
        raise ZeroDivisionError("a zero quaternion (0 + 0i + 0j + 0k) cannot be inverted")


@jit_hardcore
def quaternion_real(q: np.ndarray):
    """Return real part of quaternion.

    >>> quaternion_real(np.array(3, 0, 1, 2),dtype=float))
    3.0

    """
    return float(q[0])


@jit_hardcore
def quaternion_imag(q: np.ndarray):
    """Return imaginary part of quaternion.

    >>> quaternion_imag(np.array((3, 0, 1, 2),dtype=float))
    array([ 0.,  1.,  2.])

    """
    return np.copy(q[1:4])


@jit_hardcore
def quaternion_mul_matrix(q: np.ndarray):
    """Matrix representation of quaternion for multiplication purposes.
    """
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]])


@jit_hardcore
def quaternion_mul_bar_matrix(q):
    """Matrix representation of quaternion for multiplication purposes.
    """
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], q[3], -q[2]],
        [q[2], -q[3], q[0], q[1]],
        [q[3], q[2], -q[1], q[0]]])


@jit_hardcore
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> quaternion:
    return np.dot(quaternion_mul_matrix(q1), q2)


@jit_hardcore
def quaternion_from_scalar_and_vector(scalar: double = 0.0, vector: np.ndarray = origin) -> quaternion:
    q = np.zeros(4, dtype=double)
    q[0] = scalar
    q[1:] = vector
    return q


# @jit_hardcore
def quaternion_between_vectors(v1: np.ndarray, v2: np.ndarray) -> quaternion:
    """
    Construct rotation quaternion to rotate direction given by v1 towards v2

    :param v1: "source" direction
    :param v2: "destination" direction
    :return: Quaternion of rotation, such that ret.rotate(v1) == v2
    """

    v1 = vector_normalize(v1)
    v2 = vector_normalize(v2)
    s = v1 + v2
    n = norm(s)
    if n < EPS:
        return quaternion_from_scalar_and_vector(scalar=0, vector=vector_normalize(orthogonal(v1)))

    half = s / n
    # noinspection PyTypeChecker
    return quaternion_from_scalar_and_vector(scalar=np.dot(v1, half), vector=np.cross(v1, half))


@jit_hardcore
def quaterion_vector_conjugate(q: quaternion) -> quaternion:
    z = np.zeros(4, dtype=double)
    z[0] = q[0]
    z[1:4] = -q[1:4]
    return z


@jit_hardcore
def quaternion_normalize(q: quaternion) -> quaternion:
    """Object is guaranteed to be a unit quaternion after calling this
    operation UNLESS the object is equivalent to Quaternion(0)
    """
    n = norm(q)
    if n > 0:
        return q / n
    else:
        raise ZeroDivisionError("Can not normalize null quaternion!")


@jit_hardcore
def rotate_quaternion(rotation: quaternion, q: quaternion, assume_normalized=False) -> quaternion:
    """Rotate a quaternion vector q using the rotation.

    Params:
        q: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

    Returns:
        A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
    """
    if not assume_normalized:
        rotation = quaternion_normalize(rotation)

    r = quaternion_multiply(rotation, q)
    r = quaternion_multiply(r, quaterion_vector_conjugate(rotation))
    return r


def quaternion_axis(q, undefined=origin):
    """Get the axis or vector about which the quaternion rotation occurs

    For a null rotation (a purely real quaternion), the rotation angle will
    always be `0`, but the rotation axis is undefined.
    It is by default assumed to be `[0, 0, 0]`.

    Params:
        undefined: [optional] specify the axis vector that should define a null rotation.
            This is geometrically meaningless, and could be any of an infinite set of vectors,
            but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.

    Returns:
        A Numpy unit 3-vector describing the Quaternion object's axis of rotation.

    Note:
        This feature only makes sense when referring to a unit quaternion.
        Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not
         already one.
    """
    tolerance = 1e-17
    q = quaternion_normalize(q)
    v = q[1:4]
    n = norm(v)
    if n < tolerance:
        # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
        return undefined
    else:
        return v / n


@jit_hardcore
def derivative(q, rate: np.ndarray):
    """Get the instantaneous quaternion derivative representing a quaternion rotating at a 3D rate vector `rate`

    Params:
        rate: np 3-array (or array-like) describing rotation rates about the global x, y and z axes respectively.

    Returns:
        A unit quaternion describing the rotation rate
    """
    a = quaternion_from_scalar_and_vector(0.5)
    b = quaternion_multiply(a, q)
    c = quaternion_multiply(b, quaternion_from_scalar_and_vector(0.0, rate))
    return c


@jit_hardcore
def quaternion_integrate(q, rate: np.ndarray, timestep: double):
    """Advance a time varying quaternion to its value at a time `timestep` in the future.

    The Quaternion object will be modified to its future value.
    It is guaranteed to remain a unit quaternion.

    Params:

    rate: np 3-array (or array-like) describing rotation rates about the
        global x, y and z axes respectively.
    timestep: interval over which to integrate into the future.
        Assuming *now* is `T=0`, the integration occurs over the interval
        `T=0` to `T=timestep`. Smaller intervals are more accurate when
        `rate` changes over time.

    Note:
        The solution is closed form given the assumption that `rate` is constant
        over the interval of length `timestep`.
    """
    q = quaternion_normalize(q)

    rotation_vector = rate * timestep
    rotation_norm = norm(rotation_vector)
    if rotation_norm > 0:
        axis = rotation_vector / rotation_norm
        angle = rotation_norm
        q2 = quaternion_from_axis_angle(axis=axis, angle=angle)
        return quaternion_normalize(quaternion_multiply(q, q2))
    else:
        raise ValueError("Rotation norm is null!")


@jit_hardcore
def quaternion_slerp(q0, q1, amount=0.5):
    """Spherical Linear Interpolation between quaternions.
    Implemented as described in https://en.wikipedia.org/wiki/Slerp

    Find a valid quaternion rotation at a specified distance along the
    minor arc of a great circle passing through any two existing quaternion
    endpoints lying on the unit radius hypersphere.

    This is a class method and is called as a method of the class itself rather than on a particular instance.

    Params:
        q0: first endpoint rotation as a Quaternion object
        q1: second endpoint rotation as a Quaternion object
        amount: interpolation parameter between 0 and 1. This describes the linear placement position of
            the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`.
            Defaults to the midpoint (0.5).

    Returns:
        A new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion.

    Note:
        This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius
        hypersphere).
        Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already
             unit length.

    >>> _q0 = random_quaternion()
    >>> _q1 = random_quaternion()
    >>> q = quaternion_slerp(_q0, _q1, 0)
    >>> np.allclose(q, _q0)
    True
    >>> q = quaternion_slerp(_q0, _q1, 1)
    >>> np.allclose(q, _q1)
    True
    >>> q = quaternion_slerp(_q0, _q1, 0.5)
    >>> ang = acos(np.dot(_q0, q))
    >>> np.allclose(2, acos(np.dot(_q0, _q1)) / ang) or np.allclose(2, acos(-np.dot(_q0, _q1)) / ang)
    True
    """
    # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)
    amount = np.clip(amount, 0.0, 1.0)

    dot = np.dot(q0, q1)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    if dot < 0.0:
        q0 = -q0
        dot = -dot

    # sin_theta_0 can not be zero
    if dot > 0.9995:
        qr = q0 + amount * (q1 - q0)
        return quaternion_normalize(qr)

    theta_0 = acos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = sin(theta_0)

    theta = theta_0 * amount
    sin_theta = sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quaternion_normalize((s0 * q0) + (s1 * q1))


@jit_hardcore
def quaternion_slerp_old(quat0: np.ndarray, quat1: np.ndarray, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.

    >>> _q0 = random_quaternion()
    >>> _q1 = random_quaternion()
    >>> q = quaternion_slerp(_q0, _q1, 0)
    >>> np.allclose(q, _q0)
    True
    >>> q = quaternion_slerp(_q0, _q1, 1, 1)
    >>> np.allclose(q, _q1)
    True
    >>> q = quaternion_slerp(_q0, _q1, 0.5)
    >>> ang = acos(np.dot(_q0, q))
    >>> np.allclose(2, acos(np.dot(_q0, _q1)) / ang) or np.allclose(2, acos(-np.dot(_q0, _q1)) / ang)
    True

    """
    q0 = vector_normalize(quat0)
    q1 = vector_normalize(quat1)
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = acos(d) + spin * pi
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / sin(angle)
    q0 *= sin((1.0 - fraction) * angle) * isin
    q1 *= sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_angle(q: np.ndarray):
    """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis.

    This is guaranteed to be within the range (-pi:pi) with the direction of
    rotation indicated by the sign.

    When a particular rotation describes a 180 degree rotation about an arbitrary
    axis vector `v`, the conversion to axis / angle representation may jump
    discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`,
    each being geometrically equivalent (see Note in documentation).

    Returns:
        A real number in the range (-pi:pi) describing the angle of rotation
            in radians about a Quaternion object's axis of rotation.

    Note:
        This feature only makes sense when referring to a unit quaternion.
        Calling this method will implicitly normalise the Quaternion object to a unit quaternion if
         it is not already one.
    """
    q = quaternion_normalize(q)
    n = norm(q[1:4])
    return wrap_angle(2.0 * atan2(n, q[0]))


@jit_hardcore
def rotate_vector(rotation: quaternion, vector: np.ndarray, assume_normalized=False):
    """Rotate a 3D vector by the rotation stored in the Quaternion object.

    :param rotation: quaternion to use
    :param assume_normalized: assume rotation is unit quaternion
    :param vector: A 3-vector specified as numpy array

    :returns: The rotated vector returned as numpy array
    """
    q1 = quaternion_from_scalar_and_vector(scalar=0.0, vector=vector)

    return rotate_quaternion(rotation, q1, assume_normalized=assume_normalized)[1:4]


@jit_hardcore
def quaternion_from_axis_angle(axis: np.ndarray, angle: double):
    """Initialise from axis and angle representation

    Create a Quaternion by specifying the 3-vector rotation axis and rotation
    angle (in radians) from which the quaternion's rotation should be created.

    Params:
        axis: a valid np 3-vector
        angle: a real valued angle in radians
    """
    q = np.empty(4, dtype=double)
    mag_sq = np.dot(axis, axis)
    if mag_sq == 0.0:
        raise ValueError("Axis is a null vector!")
        # return _from_scalar_and_vector(scalar=0.0)
    # Ensure axis is in unit vector form
    if abs(1.0 - mag_sq) > EPS:
        axis = axis / sqrt(mag_sq)
    theta = angle / 2.0
    q[0] = cos(theta)
    q[1:4] = axis * sin(theta)
    return q


@jit_hardcore
def random_quaternion(randv: np.ndarray = None):
    """Generate a random unit quaternion. Supply randv as uniform (0..1) vector if predictable random is needed

    :param randv: array like or None Three independent random variables that are uniformly distributed between 0 and 1.
    :returns Unit quaternion uniformly distributed across the rotation space

    As per: http://planning.cs.uiuc.edu/node198.html
    >>> q1 = random_quaternion()
    >>> q2 = random_quaternion()
    >>> np.allclose(q1,q2)
    False
    >>> q = random_quaternion()
    >>> np.allclose(1, norm(q))
    True

    """
    if randv is None:
        r1, r2, r3 = np.random.random(3)
    else:
        r1, r2, r3 = randv
    rr1 = sqrt(1.0 - r1)
    rr2 = sqrt(r1)

    t1 = tau * r1
    t2 = tau * r2
    return np.array((cos(t2) * rr2, sin(t1) * rr1, cos(t1) * rr1, sin(t2) * rr2), dtype=double)


@jit_hardcore
def yaw_pitch_roll(q):
    """Get the equivalent yaw-pitch-roll angles aka. intrinsic Tait-Bryan angles following the z-y'-x'' convention

    Returns:
        yaw:    rotation angle around the z-axis in radians, in the range `[-pi, pi]`
        pitch:  rotation angle around the y'-axis in radians, in the range `[-pi/2, -pi/2]`
        roll:   rotation angle around the x''-axis in radians, in the range `[-pi, pi]`

    The resulting rotation_matrix would be R = R_x(roll) R_y(pitch) R_z(yaw)

    Note:
        This feature only makes sense when referring to a unit quaternion. Calling this method will implicitly
        normalise the Quaternion object to a unit quaternion if it is not already one.
    """

    q = quaternion_normalize(q)
    yaw = atan2(2 * (q[0] * q[3] - q[1] * q[2]),
                1 - 2 * (q[2] ** 2 + q[3] ** 2))
    pitch = asin(2 * (q[0] * q[2] + q[3] * q[1]))
    roll = atan2(2 * (q[0] * q[1] - q[2] * q[3]),
                 1 - 2 * (q[1] ** 2 + q[2] ** 2))

    return yaw, pitch, roll
