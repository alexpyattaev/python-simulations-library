import os
import pickle
import typing
from typing import List

import numpy as np
from matplotlib import cycler
from scipy.signal import get_window
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.figure
import matplotlib.cm as cm


def matplotlib_IEEE_style():
    """IEEE style by Olga Galinina. Applies to all figures that will be created"""
    mpl.rc('figure', facecolor='0.85')
    mpl.rc('axes', edgecolor='(0.04, 0.14, 0.42)',
           facecolor='(0.87, 0.92, 0.98)',
           labelsize='medium',
           labelweight='bold',
           labelcolor='(0.04, 0.14, 0.42)',
           grid='True')
    mpl.rc('text', color='(0.04, 0.14, 0.42)')
    mpl.rc('font', family='sans-serif', weight='bold', size=12)
    mpl.rc('legend', fontsize='medium')
    mpl.rc('axes', prop_cycle=cycler(color=[
        (0.04, 0.52, 0.78),
        (0.01, 0.57, 0.58),
        (1, 0, 0.6)]))


def matplotlib_WINTER_style():
    """WINTER style by Alex Pyattaev. Applies to all figures that will be created"""
    mpl.rc('figure', facecolor='1',
           titleweight='bold',  # weight of the figure title
           figsize=(9, 6))  # figure size in inches)
    mpl.rc('axes', edgecolor='(0.04, 0.14, 0.42)',
           # facecolor='(0.87, 0.92, 0.98)',
           labelsize='large',
           labelweight='bold',
           # labelcolor='(0.04, 0.14, 0.42)',
           grid='True')
    # mpl.rc('text', color='(0.04, 0.14, 0.42)')
    mpl.rc('lines', linewidth=2, markersize=9)

    mpl.rc('font', family='Sans', weight='bold', size=12)
    mpl.rc('legend', fontsize='medium')

    mpl.rc('axes', prop_cycle=cycler(color=['blue', 'green', 'red'], marker=('v', 'o', '+')))
    mpl.rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})


def mpl_figure(title: str, xlabel: str=None, ylabel: str=None) -> mpl.figure.Figure:
    import matplotlib.pyplot as plt

    f = plt.figure()
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    return f


def make_colors(keys: List, cmap=None):
    if cmap is None:
        if len(keys) < 5:
            cmap = cm.brg
        else:
            cmap = cm.viridis
    COLORS = [cmap(i) for i in np.linspace(0, 1, len(keys))]

    def colors_fn(q):
        return COLORS[keys.index(q)]

    return colors_fn

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = get_window(window, window_len)

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len // 2 - 1):-(window_len // 2 + 1)]


def cdfplot(data: np.ndarray, yrange: float = 1.0):
    """
    Generate CDF plot data
    :param data: the data array
    :param yrange: range of Y values, e.g. use 100 for percents
    :return: the X and Y values to be used in plot
    """
    X = np.sort(data)
    l = X.shape[0]
    Y = np.linspace(yrange / l, yrange, l)
    return X, Y


def pdfplot(data: np.ndarray, final_samples: int = 100, Q: int = 6):
    """
    Generate PDF of the data given
    :param data: the data array (1-d)
    :param final_samples: amount of samples in output. Keep this lower than data size!
    :param Q: specifies fraction of original data span to cover on each side.
    :return: the X and Y values to be used in plot
    """
    X, Y = cdfplot(data)

    if final_samples > X.shape[0] / 2:
        raise ValueError("keep downsampling reasonable! not enough data points!")
    # Prepare to interpolate the data for uniform support
    f = interp1d(X, Y, kind="linear", fill_value=(0, 1), assume_sorted=True, bounds_error=False)

    # Find limits of support data
    xmin = X.min()
    xmax = X.max()
    span = xmax - xmin
    # make even-spaced support with reserve value for ringing
    offset = int(span / Q)
    X = np.linspace(xmin - offset, xmax + offset, Y.shape[0])

    Y = f(X)
    # plt.plot(X, Y, label='sampled')
    W = int(Y.shape[0] / final_samples * 2)
    if W % 2 == 0:
        W += 1
    # print(W)
    Y1 = smooth(Y, window='blackman', window_len=W)
    # null out borders
    Y1[Y == 0] = 0
    Y1[Y == 1] = 1
    # plt.plot(X,Y1, label='smoothed')
    # Downsample
    f = interp1d(X, Y1, kind="quadratic", fill_value=(0, 1), assume_sorted=True, bounds_error=False)
    X = np.linspace(xmin - offset, xmax + offset, final_samples)
    Y2 = f(X)
    # get gradient (i.e. derivative)
    Y2 = np.gradient(Y2, X[1] - X[0])
    # Ensure resulting area is correct
    area = np.trapz(Y2, X)
    # print(area)
    Y2 = Y2 / area
    return X, Y2


def draw_point_labels(ax, P: np.ndarray, labels: List[str] = None, **kwargs) -> None:
    """
    Draws points with labels
    :param ax: axes to use. Can be 2d or 3d, either way will work
    :param P: array of points
    :param labels:
    :param kwargs: kwargs passed to plot
    """
    if labels is None:
        labels = [f"{i}" for i in range(len(P))]

    if ax.name == "3d":
        for l, p in zip(labels, P):
            ax.text(p[0] + 0.1, p[1] + 0.1, p[2] + 0.1, l, **kwargs)
    else:
        for l, p in zip(labels, P):
            ax.text(p[0] + 0.1, p[1] + 0.1, l, **kwargs)


def plot_cylinder(ax, pos, size, **kwargs):
    """Plot a cylinder in 3d.
    :param ax: axes
    :param pos: position (3-vect)
    :param size: size (radius and height)
    """
    P = 100
    x = np.linspace(-size[0], size[0], P)
    z = np.linspace(0, size[1], P)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(size[0] - Xc ** 2)
    Xc += pos[0]
    Yc += pos[1]
    Zc += pos[2]
    # Draw parameters
    rcount = 5
    ccount = 10
    ax.plot_surface(Xc, Yc, Zc, rcount=rcount, ccount=ccount, **kwargs)
    ax.plot_surface(Xc, 2 * pos[1] - Yc, Zc, rcount=rcount, ccount=ccount, **kwargs)


def plot_line(ax, p1: np.ndarray, p2: np.ndarray, *args, **kwargs) -> None:
    """
    Plot a line between two points on axes ax.

    :param ax: axes to use. Can be 2d or 3d, either way will work
    :param p1: first point
    :param p2: second point
    :param args: args passed to plot
    :param kwargs: kwargs passed to plot
    :return: None
    """
    p = np.vstack((p1, p2))
    if ax.name == "3d":
        ax.plot(p[:, 0], p[:, 1], *args, zs=p[:, 2], **kwargs)
    else:
        ax.plot(p[:, 0], p[:, 1], *args, **kwargs)


def plot_var_thick(x, wmin: float = 0.5, wmax: float = 2, **style):
    """
    Prepare a line collection to plot with variable thickness
    :param x: coordinates of points to plot, can be 2d or 3d
    :param wmin: minimal width
    :param wmax: maximal width
    :param style: stuff passed to plot
    :return:
    """

    T, D = x.shape
    lwidths = np.linspace(wmin, wmax, T)
    # Turn the points into segment groups
    xx = x.reshape(-1, 1, D)
    segments = np.concatenate([xx[:-1], xx[1:]], axis=1)
    if D == 2:
        from matplotlib.collections import LineCollection
        return LineCollection(segments, linewidths=lwidths, **style)
    elif D == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        return Line3DCollection(segments, linewidths=lwidths, **style)
    else:
        raise NotImplementedError("unsupported dimension {}".format(D))


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


default_plot_path = "./plots"  # The place where all plots will be dropped by default


def savefig(fig: object, title: str,
            path: str = default_plot_path,
            img_formats: typing.Tuple[str, ...] = ('svg', 'png'), mplfig=False) -> None:
    """
    Save a matplotlib figure into variety of formats.

    Saves static snapshot into svg and png by default.
    In addition, the figure is pickled into .mplfig file for further interactive use
    :param fig: the figure handle (Matplotlib Figure)
    :param title: the title part of filename
    :param path: the saving prefix/directory
    :param img_formats: the formats for static rendering
    :param mplfig: Save interactive Matplotlib figure along with images
    """
    for fmt in img_formats:
        fig.savefig(os.path.join(path, title + '.' + fmt), transparent=True, bbox_inches='tight')

    if mplfig:
        fobj = open(os.path.join(path, "{}.mplfig".format(title)), 'wb')
        pickle.dump(fig, fobj)
        fobj.close()


def loadfig(fname: str) -> object:
    """
    Load a matplotlib figure from pickle file

    may not work with some figures, use with caution
    :param fname: filename to read (full path)
    :return: the figure handle
    """
    fobj = open(fname, 'rb')
    fig = pickle.load(fobj)
    fobj.close()
    return fig
