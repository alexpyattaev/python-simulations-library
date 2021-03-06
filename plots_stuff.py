import os
import pickle
from itertools import repeat, count
from typing import List, Union, Callable, Tuple, Iterable, Dict

import numpy as np
from matplotlib import cycler
from matplotlib.pyplot import subplots, show
from scipy.signal import get_window
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.figure
import matplotlib.cm as cm
import matplotlib.ticker

import mpl_toolkits
import mpl_toolkits.mplot3d

plot_opts_imshow_waterfall = dict(aspect="auto", interpolation="nearest", origin='lower')


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
    mpl.rc('errorbar', capsize=5)
    mpl.rc('axes', prop_cycle=cycler(color=['blue', 'green', 'red'], marker=('v', 'o', '+')))
    mpl.rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})


def plot_timelines(data: Dict[str, np.ndarray], colors=None, time_axis=None, fig_args=None, ax_args=None)-> mpl.figure.Figure:
    if fig_args is None:
        fig_args = {'figsize': [14, 10]}
    if ax_args is None:
        ax_args = {}
    num_lines = len(data)

    colormap = matplotlib.cm.get_cmap('jet')
    if colors is None:
        colors = [colormap(k) for k in np.linspace(0, 1, num_lines)]

    f, axes = subplots(nrows=num_lines, ncols=1, sharex="all", **fig_args)
    all_handles = []
    all_labels = []
    for ax, key, clr in zip(axes, data, colors):
        if time_axis is None:
            time_axis = np.arange(len(data[key]))
        assert len(time_axis) == len(data[key]), "All timeline lengths must agree!"
        ax.plot(data[key], label=f"{key}", color=clr, **ax_args)
        ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(nbins='auto', min_n_ticks=10))
        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels
        ax.grid()


    f.legend(all_handles, all_labels, 'right')
    return f


def mpl_figure(title: str, xlabel: str = None, ylabel: str = None) -> mpl.figure.Figure:
    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid()
    return f


def make_colors(keys: list, cmap=None) -> Callable[[object], list]:
    if cmap is None:
        if len(keys) < 5:
            # noinspection PyUnresolvedReferences
            cmap = cm.brg
        else:
            # noinspection PyUnresolvedReferences
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
    L = X.shape[0]
    Y = np.linspace(yrange / L, yrange, L)
    return X, Y


def pdfplot(data: np.ndarray, final_samples: int = 100, Q: int = 6):
    """
    Generate PDF of the data given
    :param data: the data array (1-d)
    :param final_samples: amount of samples in output. Keep this lower than data size!
    :param Q: specifies fraction of original data span to cover on each side.
    :return: the X and Y values to be used in plot
    """
    if len(data) == 0:
        raise ValueError("Data can not be empty")

    X, Y = cdfplot(data)

    if final_samples > X.shape[0] / 2:
        raise ValueError("keep downsampling reasonable! not enough data points!")
    # Prepare to interpolate the data for uniform support
    f = interp1d(X, Y, kind="linear", fill_value=(0, 1), assume_sorted=True, bounds_error=False)

    # Find limits of support data
    # noinspection PyArgumentList
    xmin = X.min()
    # noinspection PyArgumentList
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


def draw_point_labels(ax: Union[matplotlib.figure.Axes, mpl_toolkits.mplot3d.axes3d.Axes3D],
                      P: np.ndarray, labels: List[str] = None, **kwargs) -> None:
    """
    Draws points with autolabel_service
    :param ax: axes to use. Can be 2d or 3d, either way will work
    :param P: array of points
    :param labels:
    :param kwargs: kwargs passed to plot
    """
    if labels is None:
        labels = [f"{i}" for i in range(len(P))]

    for l, p in zip(labels, P):
        if isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D):
            ax.text(p[0] + 0.1, p[1] + 0.1, p[2] + 0.1, l, **kwargs)
        else:
            ax.text(p[0] + 0.1, p[1] + 0.1, l, **kwargs)


def plot_cylinder(ax: mpl_toolkits.mplot3d.axes3d.Axes3D, pos, size: List[float], **kwargs):
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


def show_matrix(*args, vmin=None, vmax=None, block=True, row_break=5, sharex='none', sharey='none',
                colormaps=repeat('viridis'),
                **kwargs) -> Tuple[matplotlib.figure.Figure, Iterable[matplotlib.figure.Axes]]:
    """

    :param args:
    :param vmin:
    :param vmax:
    :param block:
    :param row_break:
    :param sharex:
    :param sharey:
    :param colormaps:
    :param kwargs:
    :return:
    """
    kwargs.update({f"arg {i}": a for i, a in enumerate(args)})
    colormaps = iter(colormaps)

    N_cols = len(kwargs)
    N_rows = N_cols // row_break + 1
    if N_rows > 1:
        N_cols = row_break

    f, axs = subplots(N_rows, N_cols, squeeze=False, sharex=sharex, sharey=sharey)
    for i, (name, data) in enumerate(kwargs.items()):
        ndim = data.ndim
        col = i % row_break
        row = i // row_break
        ax = axs[row, col]
        ax.set_title(name)
        if ndim == 1:
            ax.plot(data, label=name)
            ax.set_ylim([vmin, vmax])
        elif ndim == 2:
            im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto', origin='lower',
                           cmap=next(colormaps))
            f.colorbar(im, ax=ax)
        else:
            print(f"Provided input {name} has number of dimensions {ndim} which is not supported")
            raise ValueError('number of dimensions not supported')
    if block:
        show(block=True)
    return f, axs


def axisEqual3D(ax: mpl_toolkits.mplot3d.axes3d.Axes3D):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


default_plot_path = "./plots"  # The place where all plots will be dropped by default


def savefig(fig: matplotlib.figure.Figure, title: str,
            path: str = default_plot_path,
            img_formats: Iterable[str] = ('svg', 'png'), mplfig=False) -> None:
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
    assert isinstance(fig, matplotlib.figure.Figure), "Invalid object passed to savefig"
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


def setticks(ax, stepsize=1) -> None:
    """
    Set ticks on axes at regular intervals. Useful for showing floorplans and other scaled image data
    :param ax: axes to work on
    :param stepsize: step size in plot units
    """
    ax.get_xaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=stepsize))
    ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=stepsize))


def draw_hexes(vertices: np.ndarray, color: str, linestyle: str, linewidth: int):
    import matplotlib.pyplot as plt
    for hex in vertices:
        for i in range(len(hex)):
            start = hex[i]
            if i + 1 == len(hex):
                end = hex[0]
            else:
                end = hex[i + 1]
            # print(f"Drawing line from {start} to {end}")
            plt.plot(np.array([start[0], end[0]]), np.array([start[1], end[1]]), color=color, linestyle=linestyle,
                     linewidth=linewidth)


def power_scale_axes(ax: matplotlib.figure.Axes, axes: str="x", scale: float=0.7) -> None:
    """

    :param ax: axes object to operate on
    :param axes: axes to manipulate scale on. can be x,y or both
    :param scale: power scale to apply

    """
    def compress(x, pow=scale):
        return np.sign(x) * (np.abs(x) ** pow)

    def decompress(x, pow=scale):
        return x ** 1 / pow
    if "x" in axes:
        ax.set_xscale('function', functions=(compress, decompress))
    if "y" in axes:
        ax.set_yscale('function', functions=(compress, decompress))
