""" Euler angle manipulation module


"""

from math import sqrt, atan2, sin, cos, pi, tau
from typing import Tuple, Union

import numpy as np

from lib.numba_opt import double, jit_hardcore
from lib.stuff import EPS, sign
from lib.transformations.transform_tools import _NEXT_AXIS, AXES2TUPLE
from lib.vectors import norm, vector

__docformat__ = 'restructuredtext en'


@jit_hardcore
def vector_to_euler(v: np.ndarray):
    """
    Return roll, pitch and yaw for given offset vector
    :param v: The offset vector in cartesian coordinates
    :return: the pitch and yaw angles to match vector. Roll is always 0
    """
    v = v / norm(v)
    yaw = np.arctan2(v[1], v[0])
    pitch = np.arcsin(v[2])
    return vector(0.0, pitch, yaw)


@jit_hardcore
def euler_from_matrix(matrix: np.ndarray, axes: str = 'sxyz') -> Tuple[float, float, float]:
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4*pi) * (np.random.random(3) - 0.5)
    >>> for _axes in ['sxyz', 'rxyz']:
    ...    R0 = euler_matrix(axes=_axes, *angles)
    ...    R1 = euler_matrix(axes=_axes, *euler_from_matrix(R0, _axes))
    ...    if not np.allclose(R0, R1): print(_axes, "failed")

    """
    firstaxis: int
    parity: int
    repetition: int
    frame: int

    sy: float
    ax: float
    ay: float
    az: float

    firstaxis, parity, repetition, frame = AXES2TUPLE(axes)

    i: int = firstaxis
    j: int = _NEXT_AXIS[i + parity]
    k: int = _NEXT_AXIS[i - parity + 1]

    M = matrix[:3, :3]
    if repetition:
        sy = sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = atan2(M[i, j], M[i, k])
            ay = atan2(sy, M[i, i])
            az = atan2(M[j, i], -M[k, i])
        else:
            ax = atan2(-M[j, k], M[j, j])
            ay = atan2(sy, M[i, i])
            az = 0.0
    else:
        sy = sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if sy > EPS:
            ax = atan2(M[k, j], M[k, k])
            ay = atan2(-M[k, i], sy)
            az = atan2(M[j, i], M[i, i])
        else:
            ax = atan2(-M[j, k], M[j, j])
            ay = atan2(-M[k, i], sy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


@jit_hardcore
def euler_matrix(ai: float = 0.0, aj: float = 0.0, ak: float = 0.0, axes: str = 'sxyz') -> np.ndarray:
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> _ai, _aj, _ak = (4*pi) * (np.random.random(3) - 0.5)
    >>> for _axes in ['sxyz', 'syzx']:
    ...    R = euler_matrix(_ai, _aj, _ak, _axes)

    """
    firstaxis: int
    parity: int
    repetition: int
    frame: int
    firstaxis, parity, repetition, frame = AXES2TUPLE(axes)

    i: int = firstaxis
    j: int = _NEXT_AXIS[i + parity]
    k: int = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M: np.ndarray = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


@jit_hardcore
def cyl_to_cart(lat: float, lon: float) -> np.ndarray:
    """
    Convert cylindrical coords (as you would expect them on a globe) to cartesian unit vector

    Basically copied from https://en.wikipedia.org/wiki/Spherical_coordinate_system

    NOTE: definition of theta is normally as declination relative to horizon, which is a massive pain in the ass.
          here we use inclination relative to equator which is more human-readable

    :param lat: latitude, aka theta*, aka elevation relative to equator, from -pi/2 to pi/2
    :param lon: longitude, aka phi, aka azimuth, can be whatever
    :return: numpy array with unit vector centered at origin
    >>> cyl_to_cart(0, pi).astype(int)
    array([-1,  0,  0])
    >>> cyl_to_cart(0, 0).astype(int)
    array([1, 0, 0])
    >>> cyl_to_cart(0, pi/2).astype(int)
    array([0, 1, 0])
    >>> cyl_to_cart(0, -pi/2).astype(int)
    array([ 0, -1,  0])
    >>> cyl_to_cart(pi/2, 0).astype(int)
    array([0, 0, 1])
    >>> cyl_to_cart(-pi/2, 0).astype(int)
    array([ 0,  0, -1])
    """
    pi2 = np.pi / 2
    if not (-pi2 <= lat <= pi2):
        raise ValueError("Latitude must be from -pi/2 to pi/2 ")
    theta = -lat + pi2
    phi = lon
    R = np.empty(3, dtype=double)
    R[0] = np.sin(theta) * np.cos(phi)
    R[1] = np.sin(theta) * np.sin(phi)
    R[2] = np.cos(theta)
    return R


@jit_hardcore
def ang_square(a):
    """Return a "squared" version of the angle, suitable for rectangular grids sometimes"""
    c = cos(a)
    s = sin(a)
    x = 0
    y = 0
    if abs(c) > 0.1:
        x = sign(c)
    if abs(s) > 0.1:
        y = sign(s)
    return x, y


@jit_hardcore
def sph2cart(R, th, phi):
    sin_th = np.sin(th)
    x = R * np.outer(np.cos(phi), sin_th)
    y = R * np.outer(np.sin(phi), sin_th)
    z = R * np.outer(np.ones_like(phi), np.cos(th))
    return x, y, z


@jit_hardcore
def cart2sph1(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-15
    th = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, th, phi


@jit_hardcore
def wrap_angle(theta: Union[double, np.ndarray]):
    """Helper method: Wrap any angle to lie within (-pi, pi]

    Odd multiples of pi are wrapped to +pi (as opposed to -pi)

    >>> wrap_angle(3*pi)
    3.141592653589793
    >>> wrap_angle(-2.5*pi)
    -1.5707963267948966
    >>> wrap_angle(np.array([3*pi, -2.5*pi]))
    array([ 3.14159265, -1.57079633])
    """
    return (((-theta + pi) % tau) - pi) * -1.0


if __name__ == "__main__":
    import doctest

    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
