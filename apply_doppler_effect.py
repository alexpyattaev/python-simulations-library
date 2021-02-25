import warnings
from typing import Union, Sized
import scipy
import scipy.interpolate
import numpy as np

from lib import speed_of_light


def apply_doppler(signal: np.ndarray, relative_speed: Union[np.ndarray, float],
                  sampling_rate: float, carrier: float = 0,
                  safe_padding_area: int = 0, signal_speed: float = speed_of_light):
    """
    Apply a doppler effect to a sampled signal sequence.
    :param signal: the sampled signal (we are assuming it is actual waveform, not baseband)
    :param relative_speed: radial speed of the doppler effect (positive means objects are closing).
                           If array is given must be same length as signal
    :param sampling_rate: sampling rate in Hz
    :param carrier: carrier frequency. 0 means that signal is in RF sampling form.
    :param signal_speed: signal propagation speed in m/s (by default speed of light)
    :param safe_padding_area: the amount of padding at the end of the signal for alignment due to expansion after doppler
    :return: signal after transforms applied. Generally it will not be same length as input.
    """
    N = len(signal)

    if isinstance(relative_speed, Sized):
        assert len(relative_speed) == N
        cumulative_dist = np.cumsum(relative_speed) / sampling_rate
    else:
        cumulative_dist = np.arange(N) * relative_speed / sampling_rate

    scale_coeff = carrier / sampling_rate if carrier else 1.0

    cumulative_dist *= scale_coeff

    time_lag = cumulative_dist / signal_speed
    sample_base = np.arange(N)
    sample_newbase = sample_base + time_lag * sampling_rate
    oldmax, newmax = sample_base.max(), sample_newbase.max()

    if oldmax - newmax > safe_padding_area:
        warnings.warn("Result is shorter than input signal!"
                      " Make sure your signal is padded properly to get full transform")

    if carrier:
        signal_new = np.zeros_like(sample_newbase, dtype=complex)
        for c, r in [(np.real(signal), 1), (np.imag(signal), 1j)]:
            sig_function = scipy.interpolate.interp1d(sample_base, c, kind=2, bounds_error=False,
                                                      fill_value=(np.NaN, np.NaN), assume_sorted=True, copy=False)
            signal_new += sig_function(sample_newbase) * r
    else:
        sig_function = scipy.interpolate.interp1d(sample_base, signal, kind=2, bounds_error=False,
                                                  fill_value=(np.NaN, np.NaN), assume_sorted=True, copy=False)
        signal_new = sig_function(sample_newbase)
    flt = np.isnan(signal_new) == 0

    return signal_new[flt]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SAMPLERATE = int(4e3)
    CARRIER = 2e9
    N_SAMPLES = 2 ** 12
    PAD = 1000
    t = np.arange(0, N_SAMPLES / SAMPLERATE, 1 / SAMPLERATE)

    base_freq_Hz = np.array([1.0e3, 1.1e3])

    s0 = np.zeros_like(t, dtype=complex)
    for f in base_freq_Hz:
        # s0 += np.sin(2 * np.pi * t * f)
        s0 += np.exp(1j * 2 * np.pi * t * f)
    s0[-PAD:] = 0
    # speed = np.linspace(-10, 20, len(s0))
    #
    speed = np.full(len(s0), fill_value=-100.0)

    # signal_speed = 300
    signal_speed = speed_of_light

    plt.figure()
    for i, f in enumerate(base_freq_Hz):
        doppler_shift = speed / signal_speed * (f + CARRIER)
        plt.plot(t, doppler_shift / 1e3, label=f"expected shift freq {i} ({f}Hz)")
    plt.ylabel("Shift amount, KHz")
    plt.xlabel("Time, s")
    plt.legend(loc='best')

    s1 = apply_doppler(s0, speed, SAMPLERATE, carrier=CARRIER, signal_speed=signal_speed, safe_padding_area=PAD)
    t1 = np.arange(0, len(s1) / SAMPLERATE, 1 / SAMPLERATE)
    print(len(s0), len(t), len(s1), len(t1))
    # plt.figure()
    #
    # plt.plot(t,sbase, label='base indices')
    # plt.plot(t,snewbase, label='new indices')
    # plt.legend(loc='best')
    f, (ax1, ax2) = plt.subplots(2, 1, sharex="all")
    ax1.set_title('time domain, original')
    ax1.plot(t, np.real(s0), 'r-', label='original I')
    ax1.plot(t, np.imag(s0), 'b-', label='original Q')
    ax1.set_title('time domain, doppler')
    ax2.plot(t1, np.real(s1), 'r-', label='doppler I')
    ax2.plot(t1, np.imag(s1), 'b-', label='doppler Q')

    ax1.legend(loc='best')
    f, (ax3) = plt.subplots(1, 1, sharex="all")
    ax3.set_title('freq domain')

    for s, l in [(s0, 'original'), (s1, 'doppler')]:
        ax3.plot(np.fft.fftshift(np.fft.fftfreq(len(s), 1 / SAMPLERATE)),
                 np.fft.fftshift(np.abs(np.fft.fft(s))), label=l)

    ax3.set_xlabel('Frequency, Hz')

    ax3.legend(loc='best')

    plt.show(block=True)
