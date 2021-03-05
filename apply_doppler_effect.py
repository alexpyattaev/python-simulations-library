import warnings
from typing import Union, Sized
import scipy
import scipy.interpolate
import numpy as np
from lib import speed_of_light


def apply_doppler(signal: np.ndarray, relative_speed: Union[np.ndarray, float],
                  sampling_rate: float, carrier: float = 0,
                  signal_speed: float = speed_of_light):
    """
    Apply a doppler effect to a sampled signal sequence.
    :param signal: the sampled signal (we are assuming it is actual waveform, not baseband)
    :param relative_speed: radial speed of the doppler effect (positive means objects are closing).
                           If array is given must be same length as signal
    :param sampling_rate: sampling rate in Hz
    :param carrier: carrier in Hz, default 0 (if carrier is 0, signal is assumed to be real valued, complex otherwise)
    :param signal_speed: signal propagation speed in m/s (by default speed of light)
    :return: signal after transforms applied. Generally it will not be same length as input.
    """

    N = len(signal)

    if isinstance(relative_speed, Sized):
        cumulative_dist = np.zeros(N, dtype=float)
        cumulative_dist[1:] = np.cumsum(relative_speed[0:N-1] / sampling_rate)
    else:
        cumulative_dist = np.arange(N, dtype=float) * (relative_speed / sampling_rate)

    time_lag = -cumulative_dist / signal_speed
    sample_base = np.arange(len(time_lag), dtype=float) + time_lag * sampling_rate
    sample_newbase = np.arange(0, sample_base[-1], 1, dtype=float)

    if signal.dtype == np.complex:
        if carrier:
            signal = signal * np.exp(-1j * 2 * np.pi * time_lag * carrier)

        signal_new = np.zeros_like(sample_newbase, dtype=complex)
        for c, r in [(np.real(signal), 1), (np.imag(signal), 1j)]:
            sig_function = scipy.interpolate.interp1d(sample_base, c, kind='linear', bounds_error=True,
                                                      fill_value=(np.NaN, np.NaN), assume_sorted=True, copy=False)
            signal_new += sig_function(sample_newbase) * r
    else:
        sig_function = scipy.interpolate.interp1d(sample_base, signal, kind='linear', bounds_error=True,
                                                  fill_value=(np.NaN, np.NaN), assume_sorted=True, copy=False)
        signal_new = sig_function(sample_newbase)

    return signal_new


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SAMPLERATE = int(10e6)
    CARRIER = 80e9
    N_SAMPLES = 2 ** 14
    PAD = 1000
    t = np.arange(0, N_SAMPLES / SAMPLERATE, 1 / SAMPLERATE)
    base_freq_Hz = np.array([-3e6, 3e6])
    #base_freq_Hz = np.array([3e6])

    s0 = np.zeros_like(t, dtype=complex)
    for f in base_freq_Hz:
        s0 += np.exp(1j * 2 * np.pi * t * f)

    s0[:100] = 0
    s0[N_SAMPLES//2-100:N_SAMPLES//2+100]=0
    s0[-100:] = 0
    speed = np.linspace(-100, -120, len(s0)+PAD)
    #speed = np.linspace(-2, 3, len(s0)+PAD)
    #speed = 100.0

    signal_speed = speed_of_light
    signal_speed = 0.4e7

    if isinstance(speed, Sized):
        plt.figure()
        for i, f in enumerate(base_freq_Hz):
            doppler_shift = speed / signal_speed * (f + CARRIER)
            plt.plot(t, doppler_shift[:N_SAMPLES] / 1e3, label=f"expected shift freq {i} ({f}Hz)")
        plt.ylabel("Shift amount, KHz")
        plt.xlabel("Time, s")
        plt.legend(loc='best')

    s1 = apply_doppler(s0, speed, SAMPLERATE, signal_speed=signal_speed, carrier=CARRIER)
    t1 = np.arange(0, len(s1))/ SAMPLERATE
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

    # from scipy import signal
    # plt.figure()
    # frequencies, times, spectrogram = signal.spectrogram(s0, SAMPLERATE)
    # plt.pcolormesh(times, frequencies, spectrogram)
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.figure()
    # frequencies, times, spectrogram = signal.spectrogram(s1, SAMPLERATE)
    # plt.pcolormesh(times, frequencies, spectrogram)
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    plt.show(block=True)
