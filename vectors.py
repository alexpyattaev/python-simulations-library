import warnings
from typing import Sequence, Tuple
from math import sqrt, copysign, log
from .numba_opt import *
import numpy as np

vector = np.array


@jit_hardcore
def vector_rot90(v):
    """rotates a 3-vector around z axis counter-clockwise"""
    v[:] = v[1], -v[0], v[2]
    return v


@jit_hardcore
def norm(v: np.ndarray) -> [np.ndarray, float]:
    return np.sqrt((v * v).sum(axis=0))


@jit_hardcore
def vector_normalize(v: np.ndarray) -> np.ndarray:
    """Limit the vector's length to 1, keeping direction intact"""
    n = norm(v)
    assert n > 0, "Can not normalize null vector!"
    return v / norm(v)


@jit_hardcore
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute distance between two points
    :param p1: N-vector point
    :param p2: N-vector point
    :return:
    """
    return norm(p1 - p2)


@jit_hardcore
def distance_multipoint(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute distance between base and multiple points. Dimensions must agree.

    :param base: Base point, N-vector
    :param points: Test locations, NxM matrix
    :return: M-vector of distances
    """
    return norm((points - base).T)


@jit_hardcore
def vector_project(v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Project point p onto vector v. This will work even if v is not normalized.

    Both vectors are assumed to start at origin.
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
    """
    a = line[0]  # line "start"
    hypot_v = p - a  # Hypotenuse vector in a triangle formed by a, p, and projection of p onto line

    d = line[1] - a  # line "direction" vector
    d = d / norm(d)  # same now normalized

    length = np.dot(hypot_v, d)  # length of projection  onto d

    proj_v = d * length  # vector pointing from a to point of projection

    n_v = hypot_v - proj_v  # A normal vector connecting the line and tx

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
    proj, n_v, length = point_to_line_projection(segment, p)
    line_length = norm(segment[0] - segment[1])

    if length <= 0:
        return norm(p - segment[0])
    elif length >= line_length:
        return norm(p - segment[1])
    else:
        return norm(n_v)
