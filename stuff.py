from bisect import bisect_left
from math import log, pi
from typing import Union

import numpy as np

from lib.numba_opt import jit_hardcore, jit

speed_of_light = 299792458 * 1.0003  # meters/sec


class Any(object):
    """
    An object that compares equal with everything
    """

    def __eq__(self, other):
        return True

    def __str__(self):
        return "*"

    def __repr__(self):
        return "*"

    def __hash__(self):
        return 0


ANY = Any()


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


@jit
def log2(x):
    """
    Returns a log2 of value while preserving type
    :param x:
    :return:
    """
    t = type(x)
    return t(log(x, 2))


def sign(x: float):
    """
    :param x: input arg, must be scalar
    :return: Returns sign of value x. If x is zero, returns zero.
    """
    if x:
        return np.copysign(1.0, x)
    else:
        return 0


@jit_hardcore
def DB2RATIO(d):
    """
    Convert a value / array of values into linear scale

    Accepts scalars and numpy arrays
    :param d: dB scale value[s]
    :return: linear scale value[s]
    """
    return 10.0 ** (d / 10.0)


@jit_hardcore
def RATIO2DB(x):
    """
    Convert a value / array of values into dB scale.

    Accepts scalars and numpy arrays
    :param x: input
    :return: dB scale of input
    """
    return 10.0 * np.log10(x)


def stub(*args, **kwargs):
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


def binary_search(array, x):
    """Locate the leftmost value exactly equal to x"""
    i = bisect_left(array, x)
    if i != len(array) and array[i] == x:
        return i
    raise ValueError


@jit_hardcore
def shannon_capacity(BW_Hz, lin_SINR):
    """Return Shannon's capacity for bandwidth BW_Hz and SNR equal to SINR"""
    return BW_Hz * np.log2(1 + lin_SINR)


@jit_hardcore
def shannon_required_lin_SINR(BW_Hz, data_rate):
    """Return required linear SNR to achieve a given rate in bps over bandwidth BW_Hz"""
    return 2 ** (data_rate / BW_Hz) - 1


@jit_hardcore
def free_space_path_loss(dist: float, frequency_Hz: float):
    return 20 * np.log10(dist) + 20 * np.log10(frequency_Hz) - 147.55


@jit_hardcore
def friis_path_loss_dB(dist: [float, np.ndarray], frequency_Hz: float, n: float = 2.0) -> Union[float, np.ndarray]:
    """Return the path loss in dB according to Friis formula.

    n may be adjusted unlike free_space_path_loss
    Normally, returned path loss will be negative (as per the textbook formula definition)

    :param dist: distance in meters. Can be array of distances.
    :param frequency_Hz: carrier
    :param n: propagation constant (normally 2)
    """
    return RATIO2DB(np.power(speed_of_light / (frequency_Hz * 4 * pi * dist), n))


@jit_hardcore
def friis_range(path_loss_dB: Union[np.ndarray, float], frequency_Hz: float, n: float = 2.0) -> Union[
    np.ndarray, float]:
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
    'HEADER':  '\033[95m', 'OKBLUE': '\033[94m', 'OKGREEN': '\033[92m',
    'WARNING': '\033[93m', 'FAIL': '\033[91m', 'END': '\033[0m'
}


def color_msg(msg, color):
    return cmdline_colors[color] + msg + cmdline_colors['END']


def color_print_error(msg):
    print(cmdline_colors["FAIL"], msg, cmdline_colors['END'])


def color_print_header(msg):
    print(cmdline_colors["HEADER"], msg, cmdline_colors['END'])


def color_print_okgreen(msg):
    print(cmdline_colors["OKGREEN"], msg, cmdline_colors['END'])


def color_print_okblue(msg):
    print(cmdline_colors["OKBLUE"], msg, cmdline_colors['END'])


def color_print_warning(msg):
    print(cmdline_colors["WARNING"], msg, cmdline_colors['END'])

