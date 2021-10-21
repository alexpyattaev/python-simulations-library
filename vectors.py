__docformat__ = 'restructuredtext en'

from math import sqrt
from typing import Sequence, Tuple

import numpy as np

from lib.numba_opt import jit_hardcore


@jit_hardcore
def vector(a: float, b: float, c: float, dtype=np.float64) -> np.ndarray:
    """Create vectors with float data type by default"""
    return np.array((a, b, c), dtype=dtype)


origin, xaxis, yaxis, zaxis = vector(0, 0, 0), vector(1, 0, 0), vector(0, 1, 0), vector(0, 0, 1)


@jit_hardcore
def orthogonal(v: np.ndarray) -> np.ndarray:
    """
    Returns a vector orthogonal to v.

    This will not check that v is not a null vector, if v is null world explodes.
    :param v:
    :return: vector orthogonal to v
    >>> v0 = vector(1.74,3.4,23.1)
    >>> v1 = orthogonal(v0)
    >>> np.dot(v0,v1)
    0.0
    """
    x: float = abs(v[0])
    y: float = abs(v[1])
    z: float = abs(v[2])

    if x < y:
        if x < z:
            other = vector(1, 0, 0)
        else:
            other = vector(0, 0, 1)
    else:
        if y < z:
            other = vector(0, 1, 0)
        else:
            other = vector(0, 0, 1)
    return np.cross(v, other)


@jit_hardcore
def norm(v: np.ndarray) -> float:
    """
    Return norm of a vector
    :param v:
    :return:
    >>> norm(np.zeros(3,dtype=float))
    0.0
    >>> np.isclose(norm(np.ones(3)), sqrt(3))
    True
    >>> v0 = np.random.random(3)
    >>> n = norm(v0)
    >>> np.allclose(n, np.linalg.norm(v0))
    True
    """
    assert v.ndim == 1
    # assert v.dtype != np.complex128
    return sqrt((v * v).sum())


@jit_hardcore
def norm_many(V: np.ndarray) -> np.ndarray:
    """Same as norm but works across multiple vectors in a matrix.

       Input shape is NxD, where N is number of vectors and D is dimensionality

        :param V: the vectors to be analyzed.
        :returns: vector of norms
        :rtype: np.ndarray

        >>> x = np.array([[-1, -2], [1, 2], [3, 4], [5, 6]], dtype=float)
        >>> all([np.isclose(norm(v), n) for v, n in zip(x, norm_many(x))])
        True
        """
    assert V.ndim == 2
    # assert V.dtype != complex
    return np.sqrt((V * V).sum(axis=1))


@jit_hardcore
def vector_norm(data: np.ndarray, axis=1, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> _ = vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm(vector(1,0,0))
    1.0

    """
    data = np.array(data, dtype=np.float64)
    if out is None:
        if data.ndim == 1:
            return np.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)
    return out


@jit_hardcore
def vector_normalize(v: np.ndarray) -> np.ndarray:
    """Limit the vector's length to 1, keeping direction intact
    :param v: the vector to be normalized
    :returns normalized vector
    :rtype np.ndarray
    >>> x=vector_normalize(vector(1,1,1))
    >>> norm(x)
    1.0
    >>> v0 = np.random.random(3)
    >>> v1 = vector_normalize(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    """
    n = norm(v)
    if n == 0:
        raise ValueError("Vector has null length")
    return v / norm(v)


@jit_hardcore
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute distance between two points
    :param p1: N-vector point
    :param p2: N-vector point
    :return: distance
    >>> d=distance(vector(1,1,1),vector(2,2,2))
    >>> np.isclose(np.sqrt(3), d)
    True
    """
    return norm(p1 - p2)


@jit_hardcore
def distance_multipoint(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute distance between base and multiple points. Dimensions must agree.

    :param base: Base point, N-vector
    :param points: Test locations, NxM matrix
    :return: M-vector of distances
    :rtype: np.ndarray
    >>> x = np.array([[-1, -2], [1, 2], [3, 4], [5, 6]],dtype=float)
    >>> p = np.zeros(2)
    >>> all([i == norm(p - v) for v, i in zip(x, distance_multipoint(p, x))])
    True
    """
    return norm_many(points - base)


@jit_hardcore
def vector_project(v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Project point p onto vector v. This will work even if v is not normalized.

    Both vectors are assumed to start at origin.
    :rtype: np.ndarray
    """
    return np.dot(p, v) / (norm(v) ** 2) * v


@jit_hardcore
def point_to_line_projection(line: Sequence[np.ndarray], p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the projection of point onto a vector. Can work with any dimensions.

    Returns byproducts of internal calculations.
    :param line: iterable with two vectors (shape 2xN)
    :param p: point to project (shape N)
    :return: the point of projection proj_p, normal vector from line to p, distance from line start to proj_p
    :rtype: Tuple[np.ndarray, np.ndarray, float]
    point_to_line_projection(np.random.rand(2, 3), np.random.rand(3))
    """
    a = line[0]  # line "start"
    b = line[1]  # line "end"
    hypot_v = p - a  # Hypotenuse vector in a triangle formed by a, p, and projection of p onto line

    d = b - a  # line "direction" vector
    d = d / norm(d)  # same now normalized

    # noinspection PyTypeChecker
    length: float = np.dot(hypot_v, d)  # length of projection  onto d

    proj_v = d * length  # vector pointing from a to point of projection

    n_v = hypot_v - proj_v  # A normal vector connecting the line and point

    proj_p = p - n_v  # Get the point of projection (i.e. where p projects onto line)
    return proj_p, n_v, length


@jit_hardcore
def distance_to_line(segment, p) -> float:
    """
    Compute shortest distance between p and any point on given segment

    :param segment: segment given by start and end points (2, N)
    :param p: point to test (N)
    :return: distance
    """
    length: float
    proj, n_v, length = point_to_line_projection(segment, p)
    line_length = norm(segment[0] - segment[1])

    if length <= 0:
        return norm(p - segment[0])
    elif length >= line_length:
        return norm(p - segment[1])
    else:
        return norm(n_v)


@jit_hardcore
def vector_reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Compute vector reflection off surface given by normal n.
    :param v: input direction vector. Should be normalized.
    :param n: surface normal direction vector. Should be normalized.
    :return: reflected vector (normalized)
    :rtype: np.ndarray
    """
    return v - 2 * np.dot(n, v) * n


unit_vector = vector_normalize


@jit_hardcore
def angle_between_vectors(v0: np.ndarray, v1: np.ndarray, directed: bool = True, axis: int = 0):
    """Return angle between vectors.

    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.

    >>> a = angle_between_vectors(vector(1, -2, 3), vector(-1, 2, -3))
    >>> np.allclose(a, np.pi)
    True
    >>> a = angle_between_vectors(vector(1, -2, 3), vector(-1, 2, -3), directed=False)
    >>> np.allclose(a, 0)
    True
    """
    dot = np.sum(v0 * v1)
    dot /= norm(v0) * norm(v1)
    return np.arccos(dot if directed else np.fabs(dot))


@jit_hardcore
def split_normal_and_tangential(N: np.ndarray, V: np.ndarray):
    """Split vector into normal and tangential component w.r.t. plane given by N
    :param N: normal vector for plane
    :param V: vector to decompose
    """
    C_n = vector_project(v=N, p=V)
    C_t = V - C_n
    return C_n, C_t


def test_split_normal_and_tangential():
    for i in range(1000):
        E = np.random.uniform(-1, 1, size=3)
        N = np.random.uniform(-1, 1, size=3)

        C_n, C_t = split_normal_and_tangential(N=N, V=E)

        # print(f"normal {C_n}, tangential {C_t}")
        assert np.isclose(norm(C_n + C_t), norm(E)), "sum of vectors must agree"
        assert np.isclose(np.dot(C_n, C_t), 0), "components are orthogonal"
        assert np.isclose(np.dot(N, C_t), 0), "Tangential component is normal to N vector"
        assert np.dot(N, C_n) == norm(C_n), "Normal component is present and is along the N vector"
