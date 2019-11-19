from typing import Iterable
import scipy
import scipy.interpolate
import numpy as np

from lib import speed_of_light


def apply_doppler(signal, relative_speed, sampling_rate, signal_speed=speed_of_light):
    """
    Apply a doppler effect to a sampled signal sequence.
    :param signal: the sampled signal
    :param relative_speed: radial speed of the doppler effect (positive means objects are closing)
    :param sampling_rate: sampling rate in Hz
    :param signal_speed: signal propagation speed in m/s (by default speed of light)
    :return:
    """
    N = len(signal)
    sample_base = np.arange(N)
    if isinstance(relative_speed, Iterable):
        cumulative_dist = np.cumsum(relative_speed) / sampling_rate
    else:
        cumulative_dist = np.arange(N) / sampling_rate * relative_speed
    time_lag = cumulative_dist / signal_speed
    print("lag",time_lag)
    sample_newbase = sample_base + time_lag * sampling_rate

    sig_function = scipy.interpolate.interp1d(sample_base, signal, kind=2, bounds_error=False,
                                              fill_value=(0, 0), assume_sorted=True, copy=False)
    signal = sig_function(sample_newbase)

    return signal  # sample_base, sample_newbase,


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SAMPLERATE = int(1e3)

    t = np.linspace(0, 1, SAMPLERATE)
    base_freq_Hz = np.array([50, 75])
    maxf = base_freq_Hz.max()
    s0 = 0
    for f in base_freq_Hz:
        s0 += np.sin(2 * np.pi * t * f)

    speed = np.linspace(-50, 100, len(s0))
    # speed = np.full_like(s0,50)

    plt.figure()
    for i, f in enumerate(base_freq_Hz):
        doppler_shift = (speed / 300 * f)
        plt.plot(t, doppler_shift, label=f"expected shift freq{i}={f}Hz")
    plt.legend(loc='best')
    plt.ylim([-maxf, +maxf])
    # s1, sbase, snewbase,= apply_doppler(s0, speed, SAMPLERATE, signal_speed=300)
    s1 = apply_doppler(s0, speed, SAMPLERATE, signal_speed=300)
    # plt.figure()
    #
    # plt.plot(t,sbase, label='base indices')
    # plt.plot(t,snewbase, label='new indices')
    # plt.legend(loc='best')
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('time domain')
    ax1.plot(t, s0, label='original')
    ax1.plot(t, s1, label='doppler')
    ax1.legend(loc='best')
    ax2.set_title('freq domain')
    ax2.plot(np.fft.fftfreq(len(s0), 1 / SAMPLERATE), np.abs(np.fft.fft(s0)), label='original')
    ax2.plot(np.fft.fftfreq(len(s1), 1 / SAMPLERATE), np.abs(np.fft.fft(s1)), label='doppler')
    ax2.set_xlabel('Frequency, Hz')

    ax2.set_xlim([-maxf * 3, maxf * 3])
    ax2.legend(loc='best')

    plt.show(block=True)
