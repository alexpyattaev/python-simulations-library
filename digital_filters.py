from typing import Tuple, Iterable, Union

import numpy as np
from scipy.special._ufuncs import expit
from scipy import signal


class Stateful_Linear_Filter:
    """A linear filter wrapper for Numpy suitable to process samples one at a time.
    Closely mimics how a real signal processor would work with the data (i.e. not as array but as sequence of floats).
    """
    def __init__(self, b: np.ndarray, a: np.ndarray):
        """
        Initialize with output of filter design (b and a arrays)
        """
        self.b = b
        self.a = a
        self._state = signal.lfilter_zi(self.b, self.a)

    def __call__(self, x: float) -> float:
        """
        Actually filter the data
        :param x: value to be filtered next (only one!)
        :return: current output of the filter
        """
        x, self._state = signal.lfilter(self.b, self.a, [x], zi=self._state)
        return x[0]


def ZOH_filter(data, actual_times, desired_times):
    """
    Crude zero-order hold filter for data.
    :param data:
    :param actual_times:
    :param desired_times:
    :return:
    """
    # print('sampling times', desired_times)
    idx = np.digitize(desired_times, actual_times) - 1
    # print('using indices', idx)
    return data[idx], idx


def symmetric_log(a: np.ndarray, base=np.e, linthresh=2, linscale=1.0):
    """
    Performs symmetric log operation on elements of a.
    :param a: input array (any shape)
    :param linthresh: linearization threshold near 0 to avoid singularity
    :param base: logarithm base
    :param linscale: linearization slope
    :return:
    """
    out = np.zeros_like(a)
    a = a.flat
    b = out.flat
    log_base = np.log(base)
    linscale_adj = (linscale / (1.0 - base ** -1))
    abs_a = np.abs(a)
    inside = abs_a <= linthresh
    outside = np.logical_not(inside)
    with np.errstate(divide="ignore", invalid="ignore"):
        b[outside] = np.sign(a[outside]) * linthresh * (linscale_adj + np.log(abs_a[outside] / linthresh) / log_base)
    b[inside] = a[inside] * linscale_adj
    return out


def linear_fit(x: np.ndarray, y: np.ndarray, mean=False) -> Tuple[float, float]:
    """
    Perform a linear least-squares fit for data given. Can give centered or normal fits.
    :param x: x positions of samples
    :param y: values of samples
    :param mean: if True, will report the center of each bin rather than arbitrary shift
    :return: slope angle, constant shift
    """
    A = np.vstack([x, np.ones_like(y)]).T
    a, c = np.linalg.lstsq(A, y, rcond=None)[0]
    if mean:
        c = np.mean(y)
    return a, c


def piecewise_linear_fit(x: np.ndarray, y: np.ndarray, pieces: int, mean=False) -> Iterable[Tuple[float, float]]:
    L = len(x)
    assert L % pieces == 0, 'data must divide into required number of pieces!'
    sl = L // pieces
    for p in np.arange(pieces) * sl:
        yield (p, p + sl), linear_fit(x[p:p + sl], y[p:p + sl], mean=mean)


def binary_transition_smooth(x, xthr, S=5.0):
    return 1 - expit(x / xthr * S - S)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0, 5, 100)
    T = 1.5
    y = binary_transition_smooth(x,T, S=5.0)
    plt.plot(x, y, label='smooth transition (regression)')
    y = x < T
    plt.plot(x, y, label='binary transition (classification)')
    plt.xlabel('Distance')
    plt.ylabel('Proximity')
    plt.legend()
    plt.show()