from typing import Tuple, Iterable, Union
import unittest
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
    """

    :param x:
    :param y:
    :param pieces:
    :param mean:
    :return:

    """

    L = len(x)
    assert L % pieces == 0, 'data must divide into required number of pieces!'
    sl = L // pieces
    for p in np.arange(pieces) * sl:
        yield (p, p + sl), linear_fit(x[p:p + sl], y[p:p + sl], mean=mean)


def binary_transition_smooth(x: Union[float, np.ndarray], xthr: float, S: float = 5.0) -> Union[float, np.ndarray]:
    """
    Makes a nice transition from 1 to 0 as x increases (sort of inverse sigmoid)
    :param x: value to map (or array)
    :param xthr: threshold value at which output is 0.5
    :param S: Shape factor
    :return: mapped value (or array)

    """
    return 1 - expit(x / xthr * S - S)


from typing import Iterable, Union

import numpy as np


def rolling_window_lastaxis(a: np.ndarray, window_len: int, skip: int = 1, readonly=True):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>
    :param readonly: if True (default) returns a view only
    :param a: array to mess with
    :param window_len: window of slicing to work with
    :param skip: how many elements to skip. Set to 1 in order to skip by 1 every time.
    """
    if window_len < 1:
        raise ValueError("`window` must be at least 1.")
    if window_len > a.shape[-1]:
        raise ValueError("`window` is too long.")
    assert skip >= 1
    shape = a.shape[:-1] + ((a.shape[-1] - window_len) // skip + 1, window_len)
    print("new shape:", shape)
    strides = list(a.strides) + [a.strides[-1]]
    #print(1000001, strides)
    strides[-2] *= skip
    #print(1000001, strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=not readonly)


def rolling_window(a: np.ndarray, window_shape: Union[int, Iterable[int]], skip: int = 1, readonly=True):
    """
    Create rolling window views into array a given window shape
    :param a: array to work on
    :param window_shape: a tuple of int defining the shape (or single int)
    :param readonly: if True (default) returns a view only
    :param skip: how many elements to skip. Set to 1 in order to skip by 1 every time. Only tested for 1-d window.
    :return: view into array
    """
    if not isinstance(window_shape, Iterable):
        return rolling_window_lastaxis(a, window_shape, skip=skip, readonly=readonly)
    for i, win in enumerate(window_shape):
        assert skip == 0, 'Untested!'
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win, skip=0, readonly=readonly)
            a = a.swapaxes(-2, i)
    return a


class TestLinearFilters(unittest.TestCase):
    def test_rolling_window(self):
        filtsize = 3
        a = np.arange(10)
        a = np.tile(a, [2, 1]).T
        print(a)

        a = np.moveaxis(a, 0, -1)
        print('before',a.shape)
        b = rolling_window(a, filtsize)
        print('after',b.shape)
        self.assertEqual(b.shape, (2, 8, 3))
        b = np.moveaxis(b, [-2, -1], [0, 1])
        print(b.shape)
        print(b[0])
        print(b[1])

        b = rolling_window(a, filtsize, skip=2)
        b = np.moveaxis(b, [-2, -1], [0, 1])
        print('after2', b.shape)
        print(b[0])
        print(b[1])

        print(b[-2])
        print(b[-1])

    @unittest.skip("Requires GUI")
    def test_binary_transition(self):
        import matplotlib.pyplot as plt
        x = np.linspace(0, 5, 100)
        T = 1.5
        y = binary_transition_smooth(x, T, S=5.0)
        plt.plot(x, y, label='smooth transition (regression)')
        y = x < T
        plt.plot(x, y, label='binary transition (classification)')
        plt.xlabel('Distance')
        plt.ylabel('Proximity')
        plt.legend()

        plt.figure()
        for S in [3, 4, 5, 6]:
            plt.plot(binary_transition_smooth(np.arange(500, dtype=float), 150, S=S), label=f'S={S}')
        plt.xlabel('Value')
        plt.legend()
        plt.ylabel('Label value')
        plt.show(block=True)
        plt.show()

    @unittest.skip("Requires GUI")
    def test_pcw_linear_fit(self):
        import matplotlib.pyplot as plt
        y = np.array([1, 4, 5, 6, 8, 9, 10, 7, 6, 5, 4, 2, 2, 7, 10, 16, 18, 23, 26, 32, 15], dtype=float)
        print(len(y))
        x = np.arange(len(y))
        pwl = piecewise_linear_fit(x, y, pieces=3, mean=True)
        plt.figure()
        plt.plot(x, y, '*')
        for i, (rng, line) in enumerate(pwl):
             x2 = np.arange(*rng)
        print(rng, line)
        y2 = (x2 - x2.max()) * line[0] + line[1]
        plt.plot(x2, y2, 'g-', label=f'piecewise linear {i}')
        plt.legend()

    @unittest.skip("Requires GUI")
    def test_linear_fit(self):
        import matplotlib.pyplot as plt
        y = np.array([1, 4, 5, 6, 8, 9, 10, 7, 6, 5, 4, 2, 2, 7, 10, 16, 18, 23, 26, 32, 15], dtype=float)
        print(len(y))
        x = np.arange(len(y))
        a, c = linear_fit(x, y, mean=True)
        print(a, c)
        plt.figure()
        plt.plot(x, y, '*')
        plt.plot(x, (x - x.max() / 2) * a + c, '-', label='linear')