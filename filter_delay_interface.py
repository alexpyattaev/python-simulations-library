import collections
from typing import Iterable

import numpy as np
from collections import deque
import warnings

from scipy import signal


class FirFilter:
    def __init__(self, taps):
        self.taps = np.array(taps)
        self.window = collections.deque([0] * len(taps), maxlen=len(taps))

    def __call__(self, x):
        output = []
        if not isinstance(x, Iterable):
            x = [x]
        for val in x:
            self.window.append(val)
            output.append(np.sum(np.array(self.window)*self.taps))
        return output


class RingBufferInterface:
    def __init__(self, buffer_size: int, init_val = 0):
        self.buffer_size = buffer_size

        self._buffer = deque([init_val]*buffer_size, maxlen=buffer_size)

    def add_value(self, val):
        self._buffer.append(val)

    def get_data(self, lag: int = 0, block_size: int = 1):
        if lag + block_size > self.buffer_size:
            print(lag+block_size, self.buffer_size)
            warnings.warn("Too large lag for buffer", RuntimeWarning)
        start_offset = self.buffer_size-(block_size+lag)
        end_offset = self.buffer_size-lag
        return list(self._buffer)[start_offset:end_offset]


if __name__ == "__main__":
    def test_func(x):
        return np.random.uniform()
        # return np.sin(np.pi*2*(x/60))

    import matplotlib.pyplot as plt
    filter_window_size = 64
    resampling_block_size = 30
    data_size = 200
    buffer = RingBufferInterface(buffer_size=filter_window_size+resampling_block_size)
    original_data = []
    sinr_data = []
    filter_output = []
    filt = FirFilter(signal.firwin(filter_window_size, 0.125))

    for i in range(data_size):
        buffer.add_value(test_func(i))
        if i % resampling_block_size == 0:
            original_data.extend(buffer.get_data(lag=0, block_size=resampling_block_size))
            sinr_data.extend(buffer.get_data(lag=filter_window_size//2, block_size=resampling_block_size))
            filter_output.extend(filt(buffer.get_data(lag=0, block_size=resampling_block_size)))

    fig, ax = plt.subplots()
    plt.plot(original_data, label='Original data')
    plt.plot(sinr_data, label='SINR data to event handler')
    plt.plot(filter_output, label="Filtered data to event handler")
    plt.legend()
    plt.show()



