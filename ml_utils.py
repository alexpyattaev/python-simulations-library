"""
ML related utility functions for data conditioning (mostly tensorflow)
"""
import itertools
import os
import unittest
from typing import Union, Iterable, Tuple, List, Dict

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import Callback

from lib.digital_filters import piecewise_linear_fit


def categorical_slopes(x: np.ndarray, y: np.ndarray, pieces: int, val_bins, slope_bins):
    pwl = piecewise_linear_fit(x, y, pieces=pieces, mean=True)
    ans_val = np.zeros(pieces, dtype=int)
    ans_slope = np.zeros(pieces, dtype=int)
    for i, (rng, line) in enumerate(pwl):
        ans_slope[i] = np.searchsorted(slope_bins, line[0])
        ans_val[i] = np.searchsorted(val_bins, line[1])
        print(f'range is {line[1]}, chosen bin {ans_val[i]}')
        print(f'slope is {line[0]}, chosen bin {ans_slope[i]}')

    return ans_val, ans_slope


def save_tensorflow_model(model: tf.keras.models.Model, path: str):
    assert path != ""
    os.makedirs(path, exist_ok=True)

    model.save_weights(os.path.join(path, "parameters.h5"))

    with open(os.path.join(path, "topology.json"), 'w') as f:
        f.write(model.to_json())

    print(f"Saved model to {path}")


def load_tensorflow_model(coeffs: str, model_skel: Union[tf.keras.models.Model, str] = None) -> tf.keras.models.Model:
    if not coeffs.endswith('.h5'):
        coeffs += '.h5'
    if isinstance(model_skel, str):
        assert model_skel.endswith('json')
        model_desc = open(model_skel, 'rb').read()
        model = tf.keras.models.model_from_json(model_desc, custom_objects=None)
    elif isinstance(model_skel, tf.keras.models.Model):
        model = model_skel
    else:
        raise TypeError()
    model.load_weights(coeffs)
    print(f"Loaded model {model.name} from {coeffs}")
    return model


class TF_Imbalance_Abort_Callback(Callback):
    """Abort training when strange patterns in TP/TN balance is found"""

    def __init__(self, monitor=('TP', 'TN'), patience=5, balance_rtol=0.5):
        Callback.__init__(self)

        self.monitor = monitor
        assert len(monitor) == 2, 'must observe exactly 2 values at a time'
        self.patience = patience
        self.balance_rtol = balance_rtol
        self.completed_epochs = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.aborted = False

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.aborted = False
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.completed_epochs += 1
        vals = [logs.get(m) for m in self.monitor]
        if any(map(lambda a: a is None, vals)):
            print(f'ImbalanceAbort conditioned on metrics {self.monitor} which are not available.'
                  f'Available metrics are: ', ','.join(list(logs.keys())))
            raise RuntimeError("Metric not found")

        total = sum(vals)
        if total and np.abs(vals[0] - vals[1]) / total > self.balance_rtol:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Performance imbalance detected, aborting training!")
            else:
                print(f"Performance imbalance detected, wait {self.patience - self.wait} epochs before abort")
        else:
            self.wait = 0


class TF_Reset_States_Callback(Callback):
    def __init__(self, batches_in_msmt: Union[int, Iterable[int]]):
        Callback.__init__(self)
        self.counter = 0
        if isinstance(batches_in_msmt, int):
            self.batches_in_msmt = itertools.cycle([batches_in_msmt])
        else:
            self.batches_in_msmt = itertools.cycle(batches_in_msmt)
        self.batches_in_current_msmt = next(self.batches_in_msmt)

    def on_train_batch_begin(self, batch, logs=None):
        if self.counter == self.batches_in_current_msmt:
            self.model.reset_states()
            self.counter = 0
            # print("CALLBACK, resetting states")
            self.batches_in_current_msmt = next(self.batches_in_msmt)
        else:
            self.counter += 1

    def on_test_batch_begin(self, batch, logs=None):
        self.on_train_batch_begin(batch, logs)

    def on_test_begin(self, logs=None):
        pass
        # print("TESTING STARTS")

    def on_train_begin(self, logs=None):
        pass
        # print("TRAINING STARTS")


def validate_convergence(history, min_epochs=3, min_loss_improv=3.0, val_loss_rtol=1.0):
    issues = {}
    epochs = len(history['loss'])
    if epochs < min_epochs:
        print(f"Model converged too quickly in {epochs} epochs, expected at least {min_epochs}")
        issues['Converged too quickly'] = epochs

    loss_start = history['loss'][0]
    loss_end = history['loss'][-1]
    loss_improv = loss_start / loss_end

    val_loss_end = history['val_loss'][-1]
    val_loss_start = history['val_loss'][0]
    val_loss_improv = val_loss_start / val_loss_end

    if loss_improv < min_loss_improv and val_loss_improv < min_loss_improv:
        print(
            f"Model performance is bad {loss_end}/{val_loss_end} at final epoch vs {loss_start}/{val_loss_start} at epoch 1")
        issues['Final loss too small'] = {'loss_start': loss_start, "loss_end": loss_end,
                                          "val_loss_end": val_loss_end, "val_loss_start": val_loss_start}

    if not np.allclose(val_loss_end, loss_end, rtol=val_loss_rtol):
        print(f"Model validation loss {val_loss_end} does not match loss {loss_end} for training set")
        issues['Divergence with validation set'] = {"val_loss_end": val_loss_end, "loss_end": loss_end}

    if issues:
        issues['epochs'] = epochs
    return issues


def calc_output_bias(all_labels, num_cat) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param all_labels:
    :param num_cat:
    :return: initial output biases, class weights
    """

    initial_bias = np.zeros(num_cat, dtype=float)
    class_weights = np.ones(num_cat, dtype=float)
    if num_cat == 1:
        all_labels = np.expand_dims(all_labels, -1)
    for ldim in range(num_cat):
        pos = np.count_nonzero(all_labels[..., ldim] >= 0.01)
        neg = np.count_nonzero(all_labels[..., ldim] < 0.001)
        initial_bias[ldim] = np.log(pos / neg)
        print(f'Label channel {ldim}: {pos} positives, {neg} negatives, Output bias {initial_bias[ldim]}')
        class_weights[ldim] = (1 / neg) * (pos + neg) / 2.0
    return initial_bias, class_weights


def plot_history(name, history: List[Dict[str, np.ndarray]], keys=None, ax=None):
    if ax is None:
        f = plt.figure(figsize=(16, 10))
        ax
    epoch, history = history
    print(epoch)
    print('Available keys:', history.keys())
    for key in keys:
        try:
            val = plt.plot(epoch, history['val_' + key],
                           '--', label=f"{name} {key} Validation")
            plt.plot(epoch, history[key], color=val[0].get_color(),
                     label=f"{name} {key} Training")
        except:
            print('could not plot key ', key)
            pass

    plt.xlabel('Training epochs')
    plt.legend()

    # plt.xlim([0, max(epoch)])
    return f


def plot_CNN_layers(model):
    # summarize filter shapes
    figs = []
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            print(layer.name)
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        n_filters = filters.shape[-1]
        # exit()
        print("n_filters", n_filters)
        fig, axes = plt.subplots(2, n_filters // 2)
        axes = axes.flatten()
        for ax, i in zip(axes, range(n_filters)):
            # get the filter
            f = filters[:, :, 0, i]
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            ax.imshow(f, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto', origin='lower')
        figs.append(fig)
    return figs


def plot_CLR_stats(h: dict, LR_bounds: list = None, ax: Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes] = None,
                   lr_color="red", loss_color="blue", lr_label: str = "learning rate", loss_label: str = "loss"):
    """
    Plots the learning stats from CLR scheduler.


    :param h: pointer to the CLR history
    :param LR_bounds: Learning rate bounds (for ylimit)
    :param ax: tuple of axes to plot on (will spawn new fig if None given)
    :return: the spawned figure (or None if ax was given)
    :param lr_color: color for learning rate curve
    :param lr_label: label for  learning rate curve
    :param loss_color: color for loss curve
    :param loss_label: label for loss curve
    :return figure(None if axes were provided), axes for LR, axes for loss.
    """
    # print(list(h.keys()))
    if ax is None:
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    else:
        f = None
        try:
            ax1, ax2 = ax
        except:
            ax1 = ax
            ax2 = ax1.twinx()

    ax1.plot(h['lr'], color=lr_color, label=lr_label)
    ax1.set_ylabel('learning rate', color="red")
    if LR_bounds is not None:
        ax1.set_ylim(*LR_bounds)
    ax2.plot(h['loss'], color=loss_color, label=loss_label)
    ax2.set_ylabel('loss', color="blue")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    return f, ax1, ax2


def stateful_LSTM_batch_reordering(samples_per_sequence: List[int], batch_size: int,
                                   reverse=False) -> np.ndarray:
    """
    Produces reordering sequence for stateful LSTM batched training in Tensorflow.

    Generally stateful LSTM networks consume inputs in sequence, for batched learning it is more complicated
    LSTM batch in Tensorflow acts as multiple threads, each eating its own portion of input data. All batches should
    be consuming the

    The specific order of input data feed is calculated here to ensure the individual batched processes get the right
     data.

    One dataset with 8 timesteps gets split into 2 interleaved sequences starting at 0 and 4
    >>>stateful_LSTM_batch_reordering([8], 2)
    array([0, 4, 1, 5, 2, 6, 3, 7])

    Two datasets in blocks of 4 timesteps each, get fed as interleaved sequences
    >>> stateful_LSTM_batch_reordering([4,4], 2)
    array([ 0,  2,  1,  3,  4,  8,  5,  9,  6, 10,  7, 11])



    >>> stateful_LSTM_batch_reordering([4,8], 4)
    array([ 0,  1,  2,  3,  4,  6,  8, 10,  5,  7,  9, 11])


    >>> stateful_LSTM_batch_reordering([4,8], 2)
    array([ 0,  2,  1,  3,  4,  8,  5,  9,  6, 10,  7, 11])


    :param samples_per_sequence: number of training samples per sequence (each sample is one BPTT slice)
    :param batch_size: number of slices in training at once
    :param reverse: if reversing sequence must be produced instead of forward sequence
    :return: numpy array of indices at which input must be sampled in order to get correct feed order.
    """
    assert len(samples_per_sequence) > 0
    assert batch_size > 1
    ans = []
    cnt = 0
    for j, sps in enumerate(samples_per_sequence):
        if not reverse:
            seq_ordering = np.arange(sps).reshape([batch_size, -1]).flatten(order='F')
        else:
            seq_ordering = np.arange(sps).reshape([-1, batch_size]).flatten(order='F')
        ans.append(seq_ordering + cnt)
        cnt += len(seq_ordering)
    tiling = np.concatenate(ans)
    return tiling


def stateful_LSTM_batch_reverse_reordering(samples_per_sequence: List[int], batch_size: int):
    """
    Reverse of stateful_LSTM_batch_reordering

    useful after training to run inference in batched mode (should be applied to model output in this case)
    :param samples_per_sequence: number of training samples per sequence (each sample is one BPTT slice)
    :param batch_size: number of slices in training at once
    :param batch_size:
    :return:
    """
    return stateful_LSTM_batch_reordering(samples_per_sequence, batch_size, True)


def time_distributed_dense_layer(n, **kwargs):
    """
    Creates a time distributed Dense layer
    :param n:
    :param kwargs:
    :return:
    """
    layer = keras.layers.Dense(n, **kwargs)
    return keras.layers.TimeDistributed(layer, name=layer.name + "_TD")


def get_layer_coeffs(layer: keras.layers.Layer):
    """Get total number of trainable parameters in a layer"""
    coeffs = sum([np.prod(jj.shape) for jj in layer.trainable_weights])
    return coeffs


class Test_ml_utils(unittest.TestCase):
    def test_output_bias(self):
        all_labels = np.stack([np.random.randint(0, 2, [10000]),
                               np.random.uniform(0, 1, [10000]) > 0.75]).T

        bias, weight = calc_output_bias(all_labels, 2)

        self.assertAlmostEqual(bias[0], 0, delta=0.1)
        self.assertAlmostEqual(bias[1], np.log(1 / 3), delta=0.1)

        self.assertAlmostEqual(weight[0], 1, delta=0.1)
        self.assertAlmostEqual(weight[1], 0.75, delta=0.1)

        all_labels = np.random.randint(0, 2, [10000])
        bias, weight = calc_output_bias(all_labels, 1)

        self.assertTrue(np.allclose(bias, 0.0, atol=0.1))
        self.assertTrue(np.allclose(weight, 1.0, atol=0.1))

    @unittest.skip("requires interaction")
    def test_slopes(self):
        y = np.array([1, 4, 5, 6, 8, 9, 10, 7, 6, 5, 4, 2, 2, 7, 10, 16, 18, 23, 26, 32, 15], dtype=float)
        x = np.arange(len(y))
        range_bins = np.linspace(0, 30, 3)[1:]
        slope_bins = np.linspace(-4, 4, 3)[1:]
        print('range bins:', range_bins)
        print('slope_bins:', slope_bins)
        cat_r, cat_sl = categorical_slopes(x, y, pieces=3, val_bins=range_bins, slope_bins=slope_bins)
        print("ranges:", cat_r)
        print("slopes:", cat_sl)

        y = np.flip(y)
        cat_r, cat_sl = categorical_slopes(x, y, pieces=3, val_bins=range_bins, slope_bins=slope_bins)
        print("ranges:", cat_r)
        print("slopes:", cat_sl)
        plt.show(block=True)


def bits_to_int(arr):
    """
    Converts bit arrays into integer arrays (better version of packbits, essentially)

    :param arr: array to convert. Last dimension is converted, others preserved. Should only contain 0 and 1 elements.
    :return: converted array of integers.
    >>> bits_to_int(np.array([[0,0],[0,1],[1,0],[1,1]]))
        array([0, 1, 2, 3])

    """
    return arr.dot(1 << np.arange(arr.shape[-1] - 1, -1, -1))


if __name__ == '__main__':
    unittest.main()
    import doctest

    doctest.testmod()
