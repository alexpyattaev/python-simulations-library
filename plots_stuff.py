import os
import pickle
from itertools import repeat, zip_longest
from typing import List, Union, Callable, Tuple, Iterable, Dict

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.ticker
import mpl_toolkits
import mpl_toolkits.mplot3d
import numpy as np
from matplotlib import cycler, ticker
from matplotlib.colors import Normalize
from matplotlib.pyplot import subplots, show
from scipy.interpolate import interp1d
from scipy.signal import get_window
from numpy import ma

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
    mpl.rc('legend', fontsize='medium', handlelength=3)
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
    mpl.rc('legend', fontsize='medium', handlelength=3)
    mpl.rc('errorbar', capsize=5)
    mpl.rc('axes', prop_cycle=cycler(color=['blue', 'green', 'red'], marker=('v', 'o', '+')))
    mpl.rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})


def plot_timelines(data: Dict[str, np.ndarray], colors=None, time_axis=None, fig_args=None,
                   ax_args=None) -> mpl.figure.Figure:
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
            cmap = cm.jet
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


def nice_cdf_plot(datas, title: str, names: Iterable = tuple(),
                  markers: Iterable = tuple(), styles: Iterable = tuple(),
                  linewidths: Iterable = tuple(),
                  colors: Iterable = tuple(), ignore_missed=False,
                  vertical_threshold_lines=(0.5, 2.0),
                  legend_outside=8,
                  xlabel='Distance, m',
                  ylabel='CDF, percent'):
    """
    Makes a nice-looking CDF plot from multiple data sets.

    :param datas:
    :param title:
    :param names:
    :param markers:
    :param styles:
    :param colors:
    :param ignore_missed:
    :param vertical_threshold_lines:
    :param xlabel:
    :param ylabel:
    :return:
    """
    if not isinstance(datas, list):
        datas = [datas]
        names = [names]

    f = matplotlib.pyplot.figure(figsize=[16, 10])
    ax = matplotlib.pyplot.gca()
    for data, tit, mrk, sty, clr, lw in zip_longest(datas, names, markers, styles, colors, linewidths):
        if data is None:
            raise ValueError(f'data and names length do not match {len(datas)}, {len(names)}')

        if not ignore_missed:
            data[data > 50] = 0
        if len(data) == 0:
            continue
        X, Y = cdfplot(data, 100)
        if title is None:
            tit = ""

        ax.plot(X, Y, color=clr, marker=mrk, linestyle=sty, label=tit, linewidth=lw)
    ax.set_xlabel(xlabel)
    ax.vlines(x=vertical_threshold_lines, ymin=0, ymax=100, label='thresholds', colors='k')
    if len(datas) <= legend_outside:
        ax.legend()
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    power_scale_axes(ax, axes="x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid()
    matplotlib.pyplot.tight_layout()
    return f


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


def draw_polygons(ax: matplotlib.figure.Axes, vertices: np.ndarray, color: str, linestyle: str = '-',
                  linewidth: int = 1):
    """Draw vertices (e.g. for cells) given array of polygon points"""
    for arr in vertices:
        ax.plot(arr[:, 0], arr[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)


def power_scale_axes(ax: matplotlib.figure.Axes, axes: str = "x", scale: float = 0.7) -> None:
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


def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.

    Taken from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    https://github.com/TheChymera/chr-helpers/blob/d05eec9e42ab8c91ceb4b4dcc9405d38b7aed675/chr_matplotlib.py
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin))
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False),
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    cm.register_cmap(cmap=newcmap)

    return newcmap


def test_shifted_colormap():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid

    biased_data = np.random.random_integers(low=-15, high=5, size=(37, 37))

    orig_cmap = matplotlib.cm.coolwarm
    shifted_cmap = remappedColorMap(orig_cmap, midpoint=0.75, name='shifted')
    shrunk_cmap = remappedColorMap(orig_cmap, start=0.15, midpoint=0.75, stop=0.85, name='shrunk')

    fig = plt.figure(figsize=(6, 6))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5,
                    label_mode="1", share_all=True,
                    cbar_location="right", cbar_mode="each",
                    cbar_size="7%", cbar_pad="2%")

    # normal cmap
    im0 = grid[0].imshow(biased_data, interpolation="none", cmap=orig_cmap)
    grid.cbar_axes[0].colorbar(im0)
    grid[0].set_title('Default behavior (hard to see bias)', fontsize=8)

    im1 = grid[1].imshow(biased_data, interpolation="none", cmap=orig_cmap, vmax=15, vmin=-15)
    grid.cbar_axes[1].colorbar(im1)
    grid[1].set_title('Centered zero manually,\nbut lost upper end of dynamic range', fontsize=8)

    im2 = grid[2].imshow(biased_data, interpolation="none", cmap=shifted_cmap)
    grid.cbar_axes[2].colorbar(im2)
    grid[2].set_title('Recentered cmap with function', fontsize=8)

    im3 = grid[3].imshow(biased_data, interpolation="none", cmap=shrunk_cmap)
    grid.cbar_axes[3].colorbar(im3)
    grid[3].set_title('Recentered cmap with function\nand shrunk range', fontsize=8)

    for ax in grid:
        ax.set_yticks([])
        ax.set_xticks([])
    if 'INTERACTIVE' in os.environ:
        plt.show()


class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0)  # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            # First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat > 0] /= abs(vmax - midpoint)
            resdat[resdat < 0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if isinstance(value, Iterable):
            val = ma.asarray(value)
            val = 2 * (val - 0.5)
            val[val > 0] *= abs(vmax - midpoint)
            val[val < 0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (value - 0.5)
            if val < 0:
                return val * abs(vmin - midpoint) + midpoint
            else:
                return val * abs(vmax - midpoint) + midpoint
