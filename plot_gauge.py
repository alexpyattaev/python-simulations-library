from typing import Tuple

from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Wedge, Rectangle


def _degree_range(n):
    start = np.linspace(0, 180, n + 1, endpoint=True)[0:-1]
    end = np.linspace(0, 180, n + 1, endpoint=True)[1::]
    mid_points = start + ((end - start) / 2.)
    return np.c_[start, end], mid_points


def _rot_text(ang):
    return np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))


def gauge(ax: Axes, bounds: Tuple[float, float], arrow: float, labels=('A', 'B', 'C'), colors='plasma', title: str = '',
          bounds_check: bool = False):
    """
    some sanity checks first

    """

    n = len(labels)

    if bounds_check and not (bounds[0] <= arrow <= bounds[1]):
        raise ValueError('Arrow position out of range')
    arrow = np.clip(arrow, *bounds)

    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in n discrete colors 
    """

    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, n)
        cmap = cmap(np.arange(n))
        colors = cmap[::, :].tolist()
    if isinstance(colors, list):
        if len(colors) == n:
            colors = colors[::-1]
        else:
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), n))

    """
    begins the plotting
    """

    ang_range, mid_points = _degree_range(n)

    labels = labels[::-1]

    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors):
        # sectors
        patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=1.0))

    [ax.add_patch(p) for p in patches]

    """
    set the labels 
    """

    for mid, lab in zip(mid_points, labels):
        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,
                horizontalalignment='center', verticalalignment='center', fontsize=14,
                fontweight='bold', rotation=_rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
    ax.add_patch(r)

    if "{}" in title:
        title = title.format(arrow)
    ax.text(0, -0.05, title, horizontalalignment='center',
            verticalalignment='center', fontsize=22, fontweight='bold')

    """
    plots the arrow now
    """

    tile = 180 / n / 2
    swing = 180 - 2 * tile
    pos = tile + swing - (arrow - bounds[0]) / (bounds[1] - bounds[0]) * swing

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')


if __name__ == "__main__":
    for d in [0, 3.5, 5]:
        fig, a = plt.subplots()
        gauge(a, bounds=(0.0, 5.0), arrow=d, colors='plasma', labels=np.linspace(0, 5, 11), title="Speed={} m/s")

    plt.show(block=True)
