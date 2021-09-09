import os
from enum import IntEnum
from math import sqrt, sin, cos, pi
from typing import Tuple, Iterable, List

import numpy as np

from guitools.dialogs import ask_user
from lib import rng
from lib.grids.src.flat_topped_hex import hex_dist, hex_rect, hex_corners, hex_center, hex_neighbours, hex_disc, \
    hex_shift

from lib.transformations.euler_angles import wrap_angle
from lib.vectors import norm

# Following advice from https://www.redblobgames.com/grids/hexagons/
__author__ = 'Alex Pyattaev'

HEX = Tuple[int, int, int]


def hexgrid_polygon(h: HEX, r: float, closed: bool = False) -> np.ndarray:
    pts = hex_corners(*h, edge_length=r)
    if closed:
        pts.append(pts[0])
    return np.array(pts)


def hexgrid(h: HEX, r: float, with_sectors=True):
    x, y, = hex_center(*h, edge_length=r)
    ang = []
    if with_sectors:
        for d in range(3):
            ang.append(wrap_angle(d * 2 / 3 * pi + pi / 6))  #  - pi / 6 for alternative hex layout
        return x, y, ang
    else:
        return x, y


def hexgrid_distance(start: HEX, dest: HEX):
    return hex_dist(*start, *dest)


def pos_sector(x, y, rmin, rmax, a):
    amin = a - pi / 3
    amax = a + pi / 3
    a = rng.uniform(amin, amax)
    r = sqrt(rng.uniform((rmin / rmax) ** 2, 1.0)) * rmax
    x += cos(a) * r
    y += sin(a) * r
    return x, y


def pos_around_hex(x1: float, y1: float, r_min: float, r_max: float, num_points: int = 1):
    """
    Generate random uniform points in a hex centered around x1, y1

    This can make infinite-ish loop if r_min is very close to r_max

    :param x1: center location (float in meters)
    :param y1: center location (float in meters)
    :param r_min: minimal radial distance from hex center (float in meters, can be zero)
    :param r_max: maximal radial distance from hex center (float in meters)
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




class HexgridCluster(IntEnum):
    ONE = 1  # Single cell
    THREE = 3  # three cells
    SEVEN = 7  # Single cell and all its neighbors
    NINE = 9
    NINETEEN = 19  # Single cell, and all its two-hop neighbors


def hexgrid_cells(cluster_size: HexgridCluster) -> List[HEX]:
    """
    Return a cluster of hex cells with specific pattern.
    :param cluster_size: the pattern to use. Use enum, please.
    :return: list of hex coordinates
    """
    if cluster_size == HexgridCluster.ONE:
        return [(0, 0, 0)]
    elif cluster_size == HexgridCluster.THREE:
        return [(0, 0, 0), (1, 0, -1), (1, -1, 0)]
    elif cluster_size == HexgridCluster.NINE:
        sh2 = (3, -3, 0)
        sh3 = (2, 0, -2)
        z = [(0, 0, 0), (1, 0, -1), (1, -1, 0),
             hex_shift(-1, 1, 0, *sh2), hex_shift(0, 1, -1, *sh2), hex_shift(-1, 2, -1, *sh2),
             hex_shift(0, 0, 0, *sh3), hex_shift(1, 0, -1, *sh3), hex_shift(1, -1, 0, *sh3)]
        return [hex_shift(*i, -2, 1, 1) for i in z]

    elif cluster_size == HexgridCluster.SEVEN:
        return [(0, 0, 0)] + hex_neighbours(0, 0, 0)
    elif cluster_size == HexgridCluster.NINETEEN:
        return list(hex_disc(0, 0, 0, 2))
    else:
        raise ValueError(f"Expected a value from {HexgridCluster}")


def test_hexgrid_cells():
    # noinspection PyTypeChecker
    cs1 = hexgrid_cells(7)
    cs2 = hexgrid_cells(HexgridCluster.SEVEN)
    assert np.all(cs1 == cs2), "Should take ints and enum"

    for cls in HexgridCluster:
        assert len(hexgrid_cells(cls)) == int(cls)

    if os.environ.get('INTERACTIVE'):
        import matplotlib.pyplot as plt
        for cls in HexgridCluster:
            cs = hexgrid_cells(cls)
            plt.figure()
            for c in cs:
                print(c)
                arr = hexgrid_polygon(c, r=1, closed=True)
                plt.plot(arr[:, 0], arr[:, 1])
                x, y, _ = hexgrid(c, r=1)
                plt.text(x - 1 / 3, y - 1 / 3, f"{c}")
            plt.title(f"{cls} cells")

        plt.show()
        assert ask_user("Did the plot look OK? Make sure that no parts of circles go outside of the grid.")


def hexgrid_in_box(box_x:float, box_y:float, r:float)->List[HEX]:
    """
    Fill a box of given size with hexgrid layout.
    :param box_x: box size in meters, x
    :param box_y:box size in meters, x
    :param r: cell radius, meters
    :return: list of hex coordinates
    """
    return list(hex_rect(0, 0, 0, int(box_x / r), int(box_y / r)))


def filter_neighbors(h0: HEX, all_hexes: Iterable[HEX]):
    """ Return indices of neighbors for a given hex (given full list in all_hexes)"""
    return [i for i, h in enumerate(all_hexes) if hexgrid_distance(h, h0) == 1]


def hexgrid_freq_reuse(grid_cells: Iterable[HEX], num_channels: int = 3, reuse_dist: int = 1):
    # TODO need a function to enforce reuse pattern onto a hexgrid layout
    grid_cells = tuple(tuple(gs) for gs in grid_cells)
    freq_alloc = np.full(len(grid_cells), -1)
    all_freqs = set(range(num_channels))
    to_allocate_cells = set(range(len(grid_cells)))

    # initiate process by manually assigning cell 0
    to_allocate_cells.remove(0)
    freq_alloc[0] = 0
    boundary_set = filter_neighbors(grid_cells[0], grid_cells)
    while boundary_set:
        i = boundary_set.pop(0)
        to_allocate_cells.remove(i)
        print(f"{i=} {to_allocate_cells=}, {boundary_set=}")

        busy_freqs = set()
        for j, other in enumerate(grid_cells):
            if i == j:
                continue
            d = hexgrid_distance(grid_cells[i], other)
            if d > reuse_dist:
                continue
            f = freq_alloc[j]
            if f < 0:
                continue
            busy_freqs.add(f)
        remaining_freqs = all_freqs - busy_freqs
        if not remaining_freqs:
            #print("Can not find allocation")
            #freq_alloc[i] = 50
            #continue
            raise RuntimeError("Can not find allocation")
        freq_alloc[i] = min(remaining_freqs)
        new_neighbors = set(filter_neighbors(grid_cells[i], grid_cells))
        new_neighbors = new_neighbors.intersection(to_allocate_cells)
        new_neighbors.difference_update(set(boundary_set))
        boundary_set.extend(new_neighbors)

    return freq_alloc


def test_hexgrid_in_box():
    import matplotlib.colors
    r = 70
    cs2 = hexgrid_in_box(3 * r, 7 * r, r)
    channels = list(matplotlib.colors.TABLEAU_COLORS.keys())
    freq_alloc3 = hexgrid_freq_reuse(cs2, num_channels=3, reuse_dist=1)
    freq_alloc7 = hexgrid_freq_reuse(cs2, num_channels=7, reuse_dist=2)
    assert len(cs2) == len(freq_alloc3)
    assert len(cs2) == len(freq_alloc7)
    if os.environ.get('INTERACTIVE'):
        import matplotlib.pyplot as plt
        plt.figure()
        for i, c in enumerate(cs2):
            freq = freq_alloc3[i]
            arr = hexgrid_polygon(c, r=r, closed=False)
            plt.fill(arr[:, 0], arr[:, 1], color=channels[freq])
            x, y, _ = hexgrid(c, r)
            plt.text(x - r / 3, y - r / 3, f"#{i} freq{freq} \n {c}")

        plt.title("Box of 200x300m with 50m cells, 3 channels")

        plt.figure()
        for i, c in enumerate(cs2):
            freq = freq_alloc7[i]
            arr = hexgrid_polygon(c, r=r, closed=False)
            plt.fill(arr[:, 0], arr[:, 1], color=channels[freq])
            x, y, _ = hexgrid(c, r)
            plt.text(x - r / 3, y - r / 3, f"#{i} freq{freq} \n {c}")

        plt.title("Box of 200x300m with 50m cells, 7 channels")
        plt.show()
        assert ask_user("Did the plot look OK? Make sure that cell reuse pattern was correct")


def test_freq_reuse():
    import matplotlib.colors
    cells = hexgrid_cells(HexgridCluster.NINE)
    channels = list(matplotlib.colors.TABLEAU_COLORS.keys())
    freq_alloc3 = hexgrid_freq_reuse(cells, num_channels=3, reuse_dist=1)

    assert len(cells) == len(freq_alloc3)
    if os.environ.get('INTERACTIVE'):
        import matplotlib.pyplot as plt
        plt.figure()
        for i, c in enumerate(cells):
            freq = freq_alloc3[i]
            arr = hexgrid_polygon(c, r=1, closed=False)
            plt.fill(arr[:, 0], arr[:, 1], color=channels[freq])
            x, y, _ = hexgrid(c, 1)
            plt.text(x - 1 / 3, y - 1 / 3, f"#{i} freq{freq} \n {c}")

        plt.title("Nine cells, 3 channels, should look sane")

        plt.show()
        assert ask_user("Did the plot look OK? Make sure that cell reuse pattern was correct")


def test_pos_around_hex():
    pos = np.array(list(pos_around_hex(0, 0, 0.0, 5, num_points=50000)))
    print(pos.shape)
    assert pos.shape

    if not os.environ.get('INTERACTIVE'):
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
