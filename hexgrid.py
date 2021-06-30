import itertools
import os
from enum import IntEnum
from math import sqrt, sin, cos, pi, ceil

import numpy as np

from debug_log import warn
from guitools.dialogs import ask_user
from lib import rng
from lib.plots_stuff import plot_line
from lib.transformations.euler_angles import wrap_angle
from lib.vectors import norm
from lib.numba_opt import jit_hardcore

__author__ = 'Alex Pyattaev'


def hexgrid_polygon(x: int, y: int, r: float, closed: bool = False) -> np.ndarray:
    pos_x, pos_y = hexgrid(x, y, r, with_sectors=False)
    pts = []
    for d in range(6):
        ang = d * pi / 3
        pts.append((pos_x + r * cos(ang), pos_y + r * sin(ang)))
    if closed:
        pts.append(pts[0])
    return np.array(pts)


def hexgrid(x, y, r, with_sectors=True):
    dx = r * 1.5
    dy = r * sin(pi / 3)
    y_shift = (x % 2) * dy
    pos_x = x * dx
    pos_y = 2 * y * dy + y_shift
    ang = []
    if with_sectors:
        for d in range(3):
            ang.append(wrap_angle(d * 2 / 3 * pi - pi / 6))
        return pos_x, pos_y, ang
    else:
        return pos_x, pos_y


def pos_sector(x, y, rmin, rmax, a):
    amin = a - pi / 3
    amax = a + pi / 3
    a = rng.uniform(amin, amax)
    r = sqrt(rng.uniform((rmin / rmax) ** 2, 1.0)) * rmax
    x += cos(a) * r
    y += sin(a) * r
    return x, y


def pos_around(x, y, rmin, rmax):
    r = np.sqrt(rng.uniform((rmin / rmax), 1.0))
    r = r * rmax
    a = rng.uniform(0, 2 * pi)
    x += cos(a) * r
    y += sin(a) * r
    return x, y


def pos_around_3d(x, y, z, rmin, rmax):
    r = rng.uniform((rmin / rmax), 1.0)
    r = r * rmax

    phi = rng.uniform(0, 2 * pi)
    theta = rng.uniform(0, pi)

    x += r * sin(theta) * cos(phi)
    y += r * sin(theta) * sin(phi)
    z += r * cos(theta)

    return x, y, z


def pos_on_cylinder(x, y, z, radius, z_delta):
    phi = rng.uniform(0, 2 * pi)

    x += radius * cos(phi)
    y += radius * sin(phi)
    z += rng.uniform(-z_delta, z_delta)

    return x, y, z


def pos_around_hex(x1, y1, r_min, r_max, num_points=1):
    """
    Generate random uniform points in a hex centered around x1, y1

    This can make infinite-ish loop if r_min is very close to r_max

    :param x1: center location
    :param y1: center location
    :param r_min: minimal radial distance from hex center (can be zero)
    :param r_max: maximal radial distance from hex center
    :param num_points: total number of points to make. Will return a generator.
    :return: generator making the points.
    """
    # 3 constant vectors forming possible basis pairs 120 degrees apart
    basis = np.array([(-1., 0),
                      (.5, sqrt(3.) / 2.),
                      (.5, -sqrt(3.) / 2.)]) * r_max

    assert r_max > 0
    assert r_max > r_min
    if num_points == 0:
        return

    while num_points > 0:
        # Choose random "sector" of 120 degrees
        q = rng.randint(0, 3)
        V = np.stack((basis[q], basis[(q + 1) % 3]))

        # Choose a random point on a square
        W = np.random.rand(2)

        # Transform the square onto the basis formed by sector in V
        val = np.dot(W, V)
        # check that we are not too close to the center
        if norm(val) > r_min:
            num_points -= 1
            yield val + (x1, y1)


def plot_cell_sectors(ax, cx, cy, r: float, angles, colors=("red", "green", "blue")):
    p0 = np.array([cx, cy])
    for a, c in zip(angles, colors):
        plot_line(ax, p0, p0 + np.array([cos(a), sin(a)]) * r, color=c, linewidth=3)


@jit_hardcore
def pos_around_square(x, y, x_side, y_side):
    origin_x = x - (x_side / 2)
    origin_y = y - (y_side / 2)
    x = origin_x + rng.uniform(0, x_side)
    y = origin_y + rng.uniform(0, y_side)
    return x, y


def hexgrid_in_box(box_x, box_y, r):
    dy = r * sin(pi / 3)
    y_shift = dy
    pos_y = 2 * dy + y_shift
    pos_init = [(i, j) for i in range(0, 1 + ceil(box_x / (1.5 * r))) for j in range(0, 2 + ceil(box_y / pos_y))]
    results = []
    for i in pos_init:
        values = hexgrid(i[0], i[1], r)
        results.append(values)
    return results


class HexgridCluster(IntEnum):
    ONE = 1  # Single cell
    THREE = 3  # three cells
    SEVEN = 7  # Single cell and all its neighbors
    NINE = 9
    NINETEEN = 19  # Single cell, and all its two-hop neighbors


def hexgrid_cells(cluster_size: HexgridCluster):
    if cluster_size == HexgridCluster.ONE:
        return np.array([(0, 0)])
    elif cluster_size == HexgridCluster.THREE:
        return np.array([(0, 0), (0, -1), (1, -1)])
    elif cluster_size == HexgridCluster.NINE:
        t1 = np.array([(0, 0), (0, -1), (-1, -1)])
        t2 = np.array([(0, -1), (1, 0), (0, 0)])
        t3 = np.array([(0, 0), (0, -1), (1, 0)])
        return np.concatenate((t1, t2 + np.array([1, -1]), t3 + np.array([-1, -2])))
    elif cluster_size == HexgridCluster.SEVEN:
        return np.array([(0, 0), (0, -1), (1, -1), (-1, -1), (-1, 0), (0, 1), (1, 0)])
    elif cluster_size == HexgridCluster.NINETEEN:
        cls = [(0, 0)]
        for cell_x in range(-2, 3):
            for cell_y in range(-2, 3):
                if abs(cell_x) + abs(cell_y) > 3 or (cell_y == 2 and abs(cell_x) == 1) or (cell_x == 0 and cell_y == 0):
                    continue
                cls.append((cell_x, cell_y))
        return np.array(cls)
    else:
        raise ValueError(f"Expected a value from {HexgridCluster}")


def test_hexgrid_cells():
    # noinspection PyTypeChecker
    cs1 = hexgrid_cells(7)
    cs2 = hexgrid_cells(HexgridCluster.SEVEN)
    assert np.all(cs1 == cs2), "Should take ints and enum"
    I = os.environ.get('INTERACTIVE')
    if I:
        import matplotlib.pyplot as plt
        # noinspection PyTypeChecker
        cs = hexgrid_cells(3)
        plt.figure()
        for c in cs:
            print(c)
            arr = hexgrid_polygon(*c, r=1, closed=True)
            plt.plot(arr[:, 0], arr[:, 1])
        plt.title("Three cells in a triangle pattern")
        # noinspection PyTypeChecker
        cs = hexgrid_cells(9)
        plt.figure()
        for i, c in enumerate(cs):
            print(c)
            arr = hexgrid_polygon(*c, r=1, closed=True)
            plt.plot(arr[:, 0], arr[:, 1])
            x, y, ang = hexgrid(*c, 1)
            plt.text(x, y, f"#{i} f{i % 3}\n {c}")
            plot_cell_sectors(plt.gca(), x, y, 0.3, ang)
        plt.title("9 cells as 3 adjacent triangles")
        plt.show()
        assert ask_user("Did the plot look OK? Make sure that no parts of circles go outside of the grid.")


def hexgrid_box(size=5):
    if size % 2 == 0:
        warn('Box hexgrid produces best results when size is odd')
    r = range(int(-(size // 2)), int((size + 1) // 2))
    return list(itertools.product(r, r))


def test_pos_around_hex():
    pos = np.array(list(pos_around_hex(0, 0, 0.0, 5, num_points=50000)))
    print(pos.shape)
    assert pos.shape
    I = os.environ.get('INTERACTIVE')
    if not I:
        return
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], '.')
    plt.axis('equal')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.title("All points plotted, as dots on figure")

    plt.figure()
    plt.hist2d(pos[:, 0], pos[:, 1], bins=np.linspace(-5, 5, 40))
    plt.colorbar()
    plt.title("All points plotted, as 2D histogram.")
    plt.show()
    assert ask_user("Did the plot look OK?")
