import itertools
from math import sqrt, sin, cos, pi, ceil

from debug_log import warn
from lib import rng
from lib.transformations.euler_angles import wrap_angle
from lib.vectors import norm
from .numba_opt import *

__author__ = 'Alex Pyattaev'

def hexgrid_polygon(x, y, r, closed=False):
    dx = r * 1.5
    dy = r * sin(pi / 3)
    y_shift = (x % 2) * dy
    pos_x = x * dx
    pos_y = 2 * y * dy + y_shift
    pts = []
    for d in range(0, 6):
        ang = d * pi / 3
        pts.append((pos_x + r * cos(ang), pos_y + r * sin(ang)))
    if closed:
        pts.append(pts[0])
    return np.array(pts)


def hexgrid(x, y, r):
    dx = r * 1.5
    dy = r * sin(pi / 3)
    y_shift = (x % 2) * dy
    pos_x = x * dx
    pos_y = 2 * y * dy + y_shift
    ang = []
    for dir in range(0, 3):
        ang.append(wrap_angle(dir * 2 / 3 * pi - pi / 6))
    return pos_x, pos_y, ang


def pos_sector(x, y, rmin, rmax, a):
    amin = a - pi / 3
    amax = a + pi / 3
    a = rng.uniform(amin, amax)
    r = sqrt(rng.uniform((rmin / rmax) ** 2, 1.0)) * rmax
    x += cos(a) * r
    y += sin(a) * r
    return (x, y)


def pos_fixed(x, y, x_shift, y_shift):
    x += x_shift
    y += y_shift
    return (x, y)


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
    assert num_points > 0

    while num_points > 0:
        # Choose random "sector" of 120 degrees
        q = rng.randint(0, 3)
        V = np.stack((basis[q], basis[(q + 1) % 3]))

        # Choose a random point on a square
        W = np.random.rand(2)

        # Transform the square onto the basis formed by sector in V
        val = np.dot(W,V)
        # check that we are not too close to the center
        if norm(val) > r_min:
            num_points -= 1
            yield val + (x1, y1)


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
    pos_init = [(i, j) for i in range(0, 1 + ceil(box_x / (1.5 * r))) for j in range(0, 2 + ceil(box_y / (pos_y)))]
    results = []
    for i in pos_init:
        values = hexgrid(i[0], i[1], r)
        results.append(values)
    return results


def hexgrid_cells(cluster_size=7):
    assert cluster_size in (1, 7, 19), "Only 1, 7 and 19 cell clusters are supported"
    if cluster_size == 1:
        return [(0, 0)]
    elif cluster_size == 7:
        return [(0, 0), (0, -1), (1, -1), (-1, -1), (-1, 0), (0, 1), (1, 0)]
    elif cluster_size == 19:
        cls = [(0, 0)]
        for cell_x in range(-2, 3):
            for cell_y in range(-2, 3):
                if abs(cell_x) + abs(cell_y) > 3 or (cell_y == 2 and abs(cell_x) == 1) or (cell_x == 0 and cell_y == 0):
                    continue
                cls.append((cell_x, cell_y))
        return cls


def hexgrid_box(size=5):
    if size % 2 == 0:
        warn('Box hexgrid produces best results when size is odd')
    r = range(int(-(size // 2)), int((size + 1) // 2))
    return list(itertools.product(r, r))


if __name__ == "__main__":
    def test_hexgrid():
        pos = ((-1, 1), (1, 1), (-2, 0), (2, 0), (-1, -2), (1, -2))
        print(hexgrid(0, 0, 0))
        for p in pos:
            print(hexgrid(p[0], p[1], 0))

    def test_pos_around_hex():
        pos = list(pos_around_hex(0, 0, 0.0, 5, num_points=50000))

        pos = np.array(pos)
        print(pos.shape)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(pos[:, 0], pos[:, 1], '.')
        plt.axis('equal')
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])

        plt.figure()
        plt.hist2d(pos[:, 0], pos[:, 1], bins=np.linspace(-5, 5, 40))
        plt.colorbar()
        plt.show()