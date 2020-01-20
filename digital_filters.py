import numpy as np


def ZOH_filter(data, actual_times, desired_times):
    #print('sampling times', desired_times)
    idx = np.digitize(desired_times, actual_times) - 1
    #print('using indices', idx)
    return data[idx], idx
