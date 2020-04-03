__author__ = 'Alex Pyattaev'
import numpy as np
from scipy import signal


def IR_to_CSI(ray_powers_dBm: np.array, ray_times_s: np.array,
              system_BW_Hz: float, carrier_freq_Hz: float,
              N_FFT: int=256, grid_scale:int=4, speed_ms: np.array = (), T_reference=None, phase_estimation_error=0.0):
    SAMPLING_RATE = int(system_BW_Hz * 2)  # Sampling rate of the model
    DOPPLER_FREQS = np.array(speed_ms, dtype=float) / 3e8 * carrier_freq_Hz 
    pulse_samples = N_FFT
    T = np.array(ray_times_s, dtype=float)
    P = np.array(ray_powers_dBm, dtype=float)

    THR = -40
    PL_reference = P.max()

    P = P - PL_reference  # normalize wrt LOS component

    # clear excess components that do not affect anything anyway
    flt = P > THR
    T = T[flt]
    P = P[flt]

    if T_reference is None:
        T_reference = T.min() + phase_estimation_error
    T = T - T_reference + phase_estimation_error # remove propagation delay

    if len(DOPPLER_FREQS):
        # FIXME: Make sure this works correctly when multiple MPCs land in same time bin!!!
        # Quantize the time to nearest sample bin
        pos = np.asarray(T * SAMPLING_RATE + pulse_samples, dtype=int)
        # Convert the powers to complex domain amplitudes for a given carrier frequency
        A = np.exp(-1j * 2 * np.pi * carrier_freq_Hz * T) * np.sqrt(10 ** (P / 10))

        # print(f'A={A}, pos={pos}')

        # it is nice to work with constant grid size
        grid_size = int(pulse_samples * grid_scale)

        # Grid size is determined by the longest propagation time, plus margins
        if T.max() * SAMPLING_RATE > grid_size * 0.7:
            raise ValueError('Maximal ToF is too long, need larger grid!')

        # Make time axis for single probe pulse
        pulse_time_axis = np.linspace(-pulse_samples / 2 / SAMPLING_RATE, pulse_samples / 2 / SAMPLING_RATE,
                                      pulse_samples, dtype=float)

        mpc_grids = np.zeros([grid_size, len(A)], dtype=np.complex)
        for pos, a, i in zip(pos, A, range(len(A))):
            mpc_grids[pos, i] = a
            doppler = DOPPLER_FREQS[i]
            W = system_BW_Hz + doppler
            pulse = system_BW_Hz / SAMPLING_RATE * np.sinc(W * pulse_time_axis)
            mpc_grids[:, i] = np.convolve(mpc_grids[:, i], pulse, 'same')

        grid = np.sum(mpc_grids, axis=1)
        FFT_grid = np.fft.fftshift(np.fft.fft(grid))
        IR_power_correction = 10 * np.log10(abs(np.sum(np.square(grid) / (system_BW_Hz / SAMPLING_RATE))))
    else:
        A = np.sqrt(10 ** (P / 10))
        sc_freqs = 2j * np.pi * np.linspace(carrier_freq_Hz-system_BW_Hz/2, carrier_freq_Hz+system_BW_Hz/2, N_FFT)
        # make phases for all components
        FFT_grid = np.sum(np.exp(-sc_freqs * T[:, np.newaxis]) * A[:, np.newaxis], axis=0)
        grid = np.fft.ifft(FFT_grid)
        IR_power_correction = abs(np.sum(np.square(grid) / (system_BW_Hz / SAMPLING_RATE)))/2
        FFT_grid /= np.sqrt(IR_power_correction)
        grid /= np.sqrt(IR_power_correction)

    PRx = PL_reference + 10 * np.log10(IR_power_correction)

    return PRx, SAMPLING_RATE, grid, FFT_grid, T_reference


# for compatibility with legacy code
CSI_sampling = IR_to_CSI


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from lib import speed_of_light
    import scipy
    resample = scipy.signal.resample
    f, [[ax1, ax2], [iax1, iax2]] = plt.subplots(2,2)

    tau = .11e-9
    t = 0.1e-9
    ant_dist_m = 0.2
    carrier_F = 2.4e9
    BW = 20e6
    N_FFT = 64
    P2 = 1

    lagvector = np.array([1, 0.7])

    def press(event):
        global t
        global ant_dist_m
        for ax in [iax1, ax1, iax2, ax2]:
            ax.clear()
            ax.set_ylim([-3, 3])


        if event is not None:
            if event.key == 'left':
                t = max(1e-9, t-tau)
            if event.key == 'right':
                t = t+tau
            if event.key == 'down':
                ant_dist_m = max(0.2, ant_dist_m - 0.2)
            if event.key == 'up':
                ant_dist_m = ant_dist_m + 0.2
        ax1.set_title(f'TOF = {t*1e9:.3f}ns ({t*speed_of_light:.3f} m), spacing={ant_dist_m:.3f} m')
        # p, sr, grid, fft, tref = CSI_sampling(np.array([1, 1]),  np.array([0, t]),
        #              system_BW_Hz=BW, carrier_freq_Hz=carrier_F, N_FFT=N_FFT , speed_ms=np.zeros(2))
        p, sr, grid, fft, tref = CSI_sampling(np.array([1, P2]), np.array([0, t]),
                                              system_BW_Hz=BW, carrier_freq_Hz=carrier_F, N_FFT=N_FFT)

        ax1.plot(np.real(fft))
        ax1.plot(np.imag(fft))
        ax1.bar(np.arange(N_FFT), np.abs(fft), linewidth=2, alpha=0.5)

        iax1.bar(np.arange(N_FFT), np.real(grid), align='edge')
        iax1.bar(np.arange(N_FFT), np.imag(grid), align='edge')
        #iax1.plot(np.arange(N_FFT), np.abs(grid))
        iax1.vlines(0, 0, 1, label='first MPC')
        iax1.vlines(t * BW*2, 0, 1, label='second MPC')

        DT = ant_dist_m/speed_of_light
        p, sr, grid, fft, tref = CSI_sampling(np.array([1, P2]), np.array([0, t]) + lagvector * DT,
                                              system_BW_Hz=BW, carrier_freq_Hz=carrier_F, N_FFT=N_FFT, T_reference=tref)

        ax2.plot(np.real(fft))
        ax2.plot(np.imag(fft))
        ax2.bar(np.arange(N_FFT), np.abs(fft), linewidth=2, alpha=0.5)

        iax2.bar(np.arange(N_FFT), np.real(grid),align='edge')
        iax2.bar(np.arange(N_FFT), np.imag(grid),align='edge')
        #iax2.plot(np.arange(N_FFT), np.abs(grid))
        iax2.vlines(DT * BW*2, 0, 1, label='first MPC')
        iax2.vlines((DT + t) * BW*2, 0, 1, label='second MPC')

        f.canvas.draw()


    f.canvas.mpl_connect('key_press_event', press)


    press(None)
    plt.show()
