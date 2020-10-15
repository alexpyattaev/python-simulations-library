import itertools
from functools import partial

import numpy as np
from nptyping import NDArray as Array
from typing import Any, Union, Callable

import scipy
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt


def interpolate_2D_complex(ifun, data, eval_x=None, eval_y=None, data_argname='z') -> Union[
                           Callable[[float, float], complex], Array[(Any, Any), complex]]:
    """
    Internal interpolator engine for complex-valued 2D data

    If either eval_x or eval_y is None, will return callable which provides interpolation.

    :param ifun: interpolation engine function
    :param data: data to work with
    :param eval_x: evaluation grid, X
    :param eval_y: evaluation grid, Y
    :param data_argname: arg name which should be used to feed data input
    :return: Interpolation function or interpolated array.
    """
    interpolator_real = ifun(**{data_argname: np.real(data)})
    interpolator_imag = ifun(**{data_argname: np.imag(data)})
    if eval_x is not None and eval_y is not None:
        return interpolator_real(eval_x, eval_y) + 1j * interpolator_imag(eval_x, eval_y)
    else:
        def _interp(xx, yy):
            return interpolator_real(xx, yy) + 1j * interpolator_imag(xx, yy)

        return _interp


def TD_interpolate_complex(samples: Array[(Any, Any), complex], sample_times: Array[(Any,), float],
                           new_time_axis: Array[(Any,), float]) -> Array[(Any, Any), complex]:
    """
    Interpolate (and, if needed, extrapolate) complex valued samples in time-domain.

    Uses linear interpolation.

    :param samples: Input array. First dimension is time, second is whatever.
    :param sample_times: Times of sample-taking (same length as first dimension of samples)
    :param new_time_axis: The new time-axis to which data should be interpolated. Any length.
    :return: the interpolated array, shape is [len(new_time_axis), samples.shape[1]]

    """
    N_FFT = samples.shape[1]
    minT = sample_times[0]
    maxT = sample_times[-1]

    bbox = [minT, maxT, 0, N_FFT]
    # Alternative engine:
    # ifun = partial(scipy.interpolate.interp2d, x=times, y=np.arange(N_FFT), z=samples.T, kind='linear',
    #   copy=False, bounds_error=False)
    if not (np.diff(sample_times) >0).all():
        print("Sample times are broken!!!")
        print(sample_times)
        plt.figure()
        plt.plot(sample_times)
        plt.show(block=True)
        raise RuntimeError("Sample times are broken!!!")
    inte = interpolate_2D_complex(
        ifun=partial(scipy.interpolate.RectBivariateSpline, x=sample_times,
                     y=np.arange(N_FFT), bbox=bbox, kx=1, ky=1), data=samples,
        eval_x=new_time_axis, eval_y=np.arange(N_FFT))

    return inte


if __name__ == "__main__":
    N_FFT = 32
    T = 64

    SETS = [
        [[[1, 0, 1], [0.6, 0.3, 0.2], [0.1, 1, 0], [0.1, 0.7, 1.1], [1, 0, 1]],
         np.array([0.1, 0.9, 1.5, 2.1, 3.9])],
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
         np.array([0, 3, 4, 5, 6, 7])]
    ]

    for samples, times in SETS:
        samples = np.array([scipy.signal.resample(s, N_FFT) for s in samples]) + 0.01
        phases = np.zeros_like(samples)
        phases = (np.atleast_2d(np.ones(N_FFT)).T * np.linspace(0, np.pi, phases.shape[0])).T
        print(phases)
        print(samples.shape)
        samples = samples * np.exp(1j * phases)

        f, ((ar1, ai1), (ar2, ai2)) = plt.subplots(2, 2)
        supsamples = np.zeros([T, N_FFT])
        ar1.imshow(np.abs(samples))
        ai1.imshow(np.angle(samples))
        maxT = round(max(times)) + 3
        inte_im = TD_interpolate_complex(samples=samples, sample_times=times,
                                         new_time_axis=np.linspace(0, maxT, T)+0.3)
        ar2.imshow(np.abs(inte_im), extent=[0, N_FFT, maxT, 0])
        ai2.imshow(np.angle(inte_im), extent=[0, N_FFT, maxT, 0])

    plt.show(block=True)
