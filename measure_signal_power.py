import numpy as np


def FD_power(S: np.ndarray, Fs: float) -> float:
    """
    Returns power given signal in frequency domain.

    Here we assume that S is the amplitude spectrum (i.e. FFT of signal straight from ADC)

    :param S: frequency-domain signal. Has to be complex.
    :param Fs: Sampling frequency in Hz
    :returns: measured power in Watts
    """
    return np.sum(np.abs(S) ** 2) / Fs / len(S)


def TD_power(s: np.ndarray, Fs: float) -> float:
    """
    Returns power given signal in time domain.

    Here we assume that signal is in volts
    :param s: time-domain signal. Can be complex or real
    :param Fs: Sampling frequency in Hz
    :returns: measured power in Watts
    """
    return np.sum(np.abs(s) ** 2) / Fs


def make_unit_power_noise(n, dtype=complex) -> np.ndarray:
    """Make a test signal with power 1 in time domain.

    :param n: length of signal to make
    :param dtype: data type (complex or float)
    """
    if dtype == complex:
        return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    else:
        return np.random.randn(n)


def test_TD_power():
    Fs = 1000
    t = np.linspace(0, 10, Fs)
    s = np.sin(50 * t) + 1j * np.sin(60 * t) + np.random.randn(len(t))
    p = TD_power(s, Fs)
    assert np.allclose(p, 2, rtol=0.1), "Power should be 0.5 + 0.5 + 1 for sines and noise respectively"


def test_FD_power():
    Fs = 1000
    t = np.linspace(0, 10, Fs)
    s = np.sin(50 * t) + 1j * np.sin(60 * t) + make_unit_power_noise(len(t))
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(s)
    # plt.show()
    S = np.fft.fft(s)
    p = FD_power(S, Fs)
    assert np.allclose(p, 2, rtol=0.1), "Power should be 0.5 + 0.5 + 1 for sines and noise respectively"
