import inspect
import os
import sys
from math import log, pi
from typing import Union, Callable, Iterable, Dict

import numpy as np
import pytest
from lib.numba_opt import jit_hardcore

# epsilon for testing whether a number is close to zero
EPS: float = np.finfo(float).eps * 4.0

try:
    from scipy.constants import constants
    speed_of_light = constants.speed_of_light / 1.0003  # meters/sec in air!!!
except ImportError:
    speed_of_light = 299792458.0 / 1.0003


class _Any(object):
    """
    An object that compares equal with everything
    """

    def __eq__(self, other):
        return True

    def __str__(self):
        return "*"

    def __repr__(self):
        return "*"


ANY = _Any() #An object that compares equal with everything


class Any_Of(object):
    """
    An object that compares equal with any one of the values supplied
    """

    def __init__(self, variants):
        self.variants = variants

    def __eq__(self, other):
        return other in self.variants

    def __str__(self):
        return "*" + str(self.variants) + "*"

    def __repr__(self):
        return "*" + str(self.variants) + "*"


cap = np.clip


class DeletedAttributeError(AttributeError):
    pass


class DeletedAttribute:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        cls_name = owner.__name__
        accessed_via = f'type object {cls_name!r}' if instance is None else f'{cls_name!r} object'
        raise DeletedAttributeError(f'attribute {self.name!r} of {accessed_via} has been deleted')


def log2_typed(x: Union[float, int]) -> Union[float, int]:
    """
    Returns a log2 of value while preserving type
    :param x:
    :return:
    """
    t = type(x)
    return t(log(x, 2))


@jit_hardcore
def sign(x: float) -> int:
    """
    :param x: input arg, must be scalar
    :return: Returns sign of value x. If x is zero, returns zero.
    >>> sign(4)
    1
    >>> sign(-1.2)
    -1
    >>> sign(0)
    0
    """
    if x:
        return int(np.copysign(1.0, x))
    else:
        return 0


@jit_hardcore
def DB2RATIO(d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert a value / array of values into linear scale

    Accepts scalars and numpy arrays
    :param d: dB scale value[s]
    :return: linear scale value[s]
    """
    return 10.0 ** (d / 10.0)


@jit_hardcore
def RATIO2DB(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert a value / array of values into dB scale.

    Accepts scalars and numpy arrays
    :param x: input
    :return: dB scale of input
    """
    return 10.0 * np.log10(x)


# noinspection PyUnusedLocal
def stub(*args, **kwargs) -> None:
    """A function that accepts any args and does absolutely nothing"""
    pass





def dic_parse(s: str, sep1: str = ' ', sep2: str = '_'):
    """Parse a string of form "A_4.3 B_3 C_0" into a python dictionary.
     No conversions are made to the values, i.e. the mapping is str to str. """
    d = {}
    try:
        fields = s.split(sep1)
        for f in fields:
            if f:
                n, v = f.split(sep2)
                d[n] = v
    finally:
        return d


@jit_hardcore
def shannon_capacity(BW_Hz: Union[float, np.ndarray], lin_SINR: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return Shannon's capacity for bandwidth BW_Hz and SNR equal to SINR"""
    return BW_Hz * np.log2(1 + lin_SINR)


@jit_hardcore
def shannon_required_lin_SINR(BW_Hz: Union[float, np.ndarray],
                              data_rate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return required linear SNR to achieve a given rate in bps over bandwidth BW_Hz"""
    return 2 ** (data_rate / BW_Hz) - 1


@jit_hardcore
def free_space_path_loss(dist: Union[float, np.ndarray], frequency_Hz: Union[float, np.ndarray]) -> float:
    return 20 * np.log10(dist) + 20 * np.log10(frequency_Hz) - 147.55


@jit_hardcore
def friis_path_loss(dist: Union[float, np.ndarray], frequency_Hz: float, n: float = 2.0) -> Union[float, np.ndarray]:
    """Return the path loss according to Friis formula.

    n may be adjusted unlike free_space_path_loss
    Normally, returned path loss will be negative (as per the textbook formula definition)

    :param dist: distance in meters. Can be array of distances.
    :param frequency_Hz: carrier
    :param n: propagation constant (normally 2)
    """
    return np.power(speed_of_light / (frequency_Hz * 4 * pi * dist), n)


@jit_hardcore
def friis_path_loss_dB(dist: Union[float, np.ndarray], frequency_Hz: float, n: float = 2.0) -> Union[float, np.ndarray]:
    """Return the path loss in dB according to Friis formula.

    n may be adjusted unlike free_space_path_loss
    Normally, returned path loss will be negative (as per the textbook formula definition)

    :param dist: distance in meters. Can be array of distances.
    :param frequency_Hz: carrier
    :param n: propagation constant (normally 2)
    """
    return RATIO2DB(np.power(speed_of_light / (frequency_Hz * 4 * pi * dist), n))


@jit_hardcore
def friis_range(path_loss_dB: Union[np.ndarray, float],
                frequency_Hz: float, n: float = 2.0) -> Union[np.ndarray, float]:
    """Return the range according to Friis formula given path loss in dB. n may be adjusted

    If the path loss is negative, an absolute value is taken.

    :param path_loss_dB: path loss in dB
    :param frequency_Hz: carrier
    :param n: propagation constant (normally 2)
    """
    return speed_of_light / (np.power(DB2RATIO(-np.abs(path_loss_dB)), 1 / n) * 4 * pi * frequency_Hz)


float_inf = float("Inf")
float_nan = float("NaN")

integer_types = (int, np.integer)

numeric_types = (float, int, np.number)

# This somewhat depends on what platform you are on.
# The most common way to do this is by printing ANSI escape sequences.
cmdline_colors = {
    'HEADER': '\033[95m', 'OKBLUE': '\033[94m', 'OKGREEN': '\033[92m',
    'WARNING': '\033[93m', 'FAIL': '\033[91m', 'END': '\033[0m'
}


def color_msg(msg, color):
    return cmdline_colors[color] + msg + cmdline_colors['END']


def color_print(kind, msg, fd=(sys.stdout,)):
    for f in fd:
        if kind is None:
            print(msg, file=f, flush=True)
        else:
            print(cmdline_colors[kind], msg, cmdline_colors['END'], file=f, flush=True)


def color_print_error(msg, fd=(sys.stdout,)):
    color_print("FAIL", msg, fd=fd)


def color_print_header(msg, fd=(sys.stdout,)):
    color_print("HEADER", msg, fd=fd)


def color_print_okgreen(msg, fd=(sys.stdout,)):
    color_print("OKGREEN", msg, fd=fd)


def color_print_okblue(msg, fd=(sys.stdout,)):
    color_print("OKBLUE", msg, fd=fd)


def color_print_warning(msg, fd=(sys.stdout,)):
    color_print("WARNING", msg, fd=fd)


def merge_axes(arr: np.ndarray, mergelist: Iterable[int]):
    """
    Merges arbitrary contiguous group of axes of a numpy array
    :param arr:
    :param mergelist:
    :return:
    >>> x = np.zeros(2 * 3 * 4 * 5 * 6,dtype=int).reshape([2, 3, 4, 5, 6])
    >>> for i in range(5):
    ...     x[0, 0, 2, i, :] = np.arange(6) + (i * 6)
    >>> for i in range(5):
    ...     x[0, 1, 0, i, :] = -(np.arange(6) + (i * 6))
    >>> y = merge_axes(x, [1, 2])
    >>> y.shape
    (2, 12, 5, 6)
    >>> y[0, 2, :, :]
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29]])
    >>> y[0, 4, :, :]
    array([[  0,  -1,  -2,  -3,  -4,  -5],
           [ -6,  -7,  -8,  -9, -10, -11],
           [-12, -13, -14, -15, -16, -17],
           [-18, -19, -20, -21, -22, -23],
           [-24, -25, -26, -27, -28, -29]])
    >>> z = merge_axes(x, [2, 3])
    >>> z.shape
    (2, 3, 20, 6)
    >>> z[0, 0, 10:15, :]
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29]])
    >>> z[0, 1, 0:5, :]
    array([[  0,  -1,  -2,  -3,  -4,  -5],
           [ -6,  -7,  -8,  -9, -10, -11],
           [-12, -13, -14, -15, -16, -17],
           [-18, -19, -20, -21, -22, -23],
           [-24, -25, -26, -27, -28, -29]])

    """
    ndim = arr.ndim
    mergelist = np.array(mergelist, dtype=int)
    assert (np.diff(mergelist) == 1).all()
    # figure out where the axes would land prior to merge
    na = np.arange(ndim - len(mergelist), ndim)
    # move the axes to their desired pos
    arr = np.moveaxis(arr, mergelist, na)
    # reshape the array, collapsing final len(mergelist) dimensions
    newshape = list(arr.shape[0:ndim - len(mergelist)]) + [-1]
    arr = arr.reshape(newshape)
    # Now we can move the collapsed axis back into its proper position
    # noinspection PyArgumentList
    arr = np.moveaxis(arr, len(newshape) - 1, mergelist.min())
    return arr


def bool_array_to_string(arr: Iterable[Union[bool, int]]) -> str:
    """

    :param arr:
    :return:

    >>> bool_array_to_string([1,0,0,1])
    '1001'

    """
    return "".join(("01"[i] for i in arr))


def fitargs(f: Callable, kwargs: Dict[str, str]) -> Dict[str, str]:
    """
    Fit keyword arguments to called function parameter list
    :param f: function to use
    :param kwargs: available dict with kwargs
    :return: filtered list of kwargs that can be used to call the function
    """
    return {k: kwargs[k] for k in inspect.signature(f).parameters if k in kwargs}


class Do_Not_Copy:
    def __copy__(self):
        raise RecursionError(f"One should not make copies of {self.__class__.__name__} object!")

    def __deepcopy__(self, memodict=None):
        raise RecursionError(f"One should not make deep copies of {self.__class__.__name__} object!")


def test_Do_Not_Copy():
    class boo(Do_Not_Copy):
        def __init__(self, x):
            self.x = x

    a = boo(4)
    from copy import copy, deepcopy
    with pytest.raises(RecursionError):
        b = deepcopy(a)
    with pytest.raises(RecursionError):
        b = copy(a)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
