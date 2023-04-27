from typing import Tuple

import numpy as np

from lib.numba_opt import jit_hardcore, double, TypedList
from lib.stuff import EPS

transform_matrix = np.ndarray


@jit_hardcore
def is_same_transform(matrix0: transform_matrix, matrix1: transform_matrix, eps=EPS) -> bool:
    """Return True if two matrices perform same transformation.

    >>> is_same_transform(np.ident(4), np.ident(4))
    True
    >>> z = np.ident(4)
    >>> z[1,2] = 23.0
    >>> is_same_transform(np.ident(4), z)
    False

    """
    matrix0 = np.copy(matrix0)
    matrix0 /= matrix0[3, 3]
    matrix1 = np.copy(matrix1)
    matrix1 /= matrix1[3, 3]
    return np.all(np.abs(matrix0 - matrix1) < eps)


@jit_hardcore
def identity_matrix() -> transform_matrix:
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> np.allclose(I, np.dot(I, I))
    True
    >>> np.sum(I), np.trace(I)
    (4.0, 4.0)
    >>> np.allclose(I, np.ident(4))
    True

    """
    return np.identity(4, double)


_NEXT_AXIS = (1, 2, 0, 1)


@jit_hardcore
def AXES2TUPLE(axes: str) -> Tuple[int, int, int, int]:
    """
    map axes strings to/from tuples of inner axis, parity, repetition, frame
    :param axes:
    :return:

    >>> AXES2TUPLE('sxyz')
    (0, 0, 0, 0)
    >>> AXES2TUPLE('ryxy')
    (1, 1, 1, 1)
    """
    LUT = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
    return LUT[axes]


def concatenate_matrices(*matrices) -> transform_matrix:
    """Return concatenation of series of transformation matrices.

    >>> _M = np.random.rand(16).reshape((4, 4)) - 0.5
    >>> np.allclose(_M, concatenate_matrices(_M))
    True
    >>> np.allclose(np.dot(_M, _M.T), concatenate_matrices(_M, _M.T))
    True

    """
    lst = TypedList()
    for i in matrices:
        assert isinstance(i, np.ndarray)
        lst.append(np.ascontiguousarray(i))

    return _concatenate_matrices(lst)


@jit_hardcore
def _concatenate_matrices(matrices: TypedList) -> transform_matrix:
    """Return concatenation of series of transformation matrices. Internal numba-only core. Input must be a TypedList
    """
    i: np.ndarray
    M: np.ndarray = np.identity(4)
    for i in matrices:
        M = np.dot(M, i)
    return M


if __name__ == "__main__":
    import doctest

    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
