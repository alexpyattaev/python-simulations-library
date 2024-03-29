from math import sqrt
from typing import Sequence, Tuple

import pytest
import scipy.special
from numpy import NaN

from lib.stuff import float_inf
import numpy as np


def get_confidence(array: Sequence, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Gets the confidence interval for array of observations.

    Uses inverse Q-function for this.
    https://en.wikipedia.org/wiki/Q-function

    :raises: ValueError on empty arrays, ArithmeticError on invalid confidence parameters
    :param array: observations
    :param confidence: desired confidence level (0..1)
    :return: mean, error bound
    """
    if not (0 <= confidence < 1.0):
        raise ArithmeticError("Can not be more than 100% certain")

    array = np.array(array)
    N = np.sum(np.isfinite(array))
    if N == 0:
        return NaN, NaN

    if N == 1:
        return array[0], float_inf

    # noinspection PyTypeChecker
    return np.nanmean(array), sqrt(2)*scipy.special.erfinv(confidence) * np.nanstd(array) / sqrt(N)


def test_get_confidence():
    arr = np.random.uniform(5, 10, 10000)
    v, c = get_confidence(arr)
    assert 7.4 < v < 7.6
    assert c < 0.1

    v, c = get_confidence(arr, confidence=0)
    assert c == 0.0

    with pytest.raises(ArithmeticError):
        get_confidence(arr, confidence=2.0)

    with pytest.raises(ArithmeticError):
        get_confidence(arr, confidence=-1)


def test_get_confidence_NAN_treatment():
    arr = np.random.uniform(5, 10, 10000)
    arr[700] = NaN
    v, c = get_confidence(arr)
    assert 7.4 < v < 7.6
    assert c < 0.1

