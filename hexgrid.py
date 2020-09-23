from math import sqrt, sin, cos, pi, atan2, floor, ceil


from lib import pi_wrap
from lib.vectors import norm
from lib import rng
import numpy as np
import itertools
from debug_log import warn

from .numba_opt import *


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
        ang.append(pi_wrap(dir * 2 / 3 * pi - pi / 6))
    return pos_x, pos_y, ang

@jit
def pos_sector(x, y, rmin, rmax, a):
    amin = a - pi / 3
    amax = a + pi / 3
    a = rng.uniform(amin, amax)
    r = sqrt(rng.uniform((rmin / rmax) ** 2, 1.0)) * rmax
    x += cos(a) * r
    y += sin(a) * r
    return (x, y)

@jit
def pos_fixed(x, y, x_shift, y_shift):
    x += x_shift
    y += y_shift
    return (x, y)

@jit
def pos_around(x, y, rmin, rmax):
    r = np.sqrt(rng.uniform((rmin / rmax), 1.0))
    r = r * rmax
    a = rng.uniform(0, 2 * pi)
    x += cos(a) * r
    y += sin(a) * r
    return x, y

@jit
def pos_around_3d(x, y, z, rmin, rmax):
    r = rng.uniform((rmin / rmax), 1.0)
    r = r * rmax

    phi = rng.uniform(0, 2*pi)
    theta = rng.uniform(0, pi)

    x += r * sin(theta) * cos(phi)
    y += r * sin(theta) * sin(phi)
    z += r * cos(theta)

    return x, y, z


@jit
def pos_on_cylinder(x, y, z, radius, z_delta):
    phi = rng.uniform(0, 2*pi)

    x += radius * cos(phi)
    y += radius * sin(phi)
    z += rng.uniform(-z_delta, z_delta)

    return x, y, z

@jit
def pos_around_hex(x1, y1, rmin, r_max):
    #FIXME: this needs rethinking
    vectors = np.array([(-1., 0), (.5, sqrt(3.) / 2.), (.5, -sqrt(3.) / 2.)]) * r_max

    ind = True
    while ind:
        x = rng.randint(0, 3)
        (v1, v2) = (vectors[x], vectors[(x + 1) % 3])
        (x, y) = (rng.rand(), rng.rand())
        val = np.array([x * v1[0] + y * v2[0], x * v1[1] + y * v2[1]])
        if norm(np.array(val)-np.array([x,y])) > rmin:
            ind = False
            val = val + (x1,y1)
    return val


@jit_hardcore
def pos_around_square(x, y, x_side, y_side):
    origin_x = x - (x_side / 2)
    origin_y = y - (y_side / 2)
    x = origin_x + rng.uniform(0, x_side)
    y = origin_y + rng.uniform(0, y_side)
    return x, y

def hexgrid_in_box(box_x,box_y,r):
    dy = r * sin(pi / 3)
    y_shift = dy
    pos_y = 2 * dy + y_shift
    pos_init = [ (i,j) for i in range(0,1+ceil(box_x/(1.5*r))) for j in range(0,2+ceil(box_y/(pos_y)))]
    results = []
    for i in pos_init:
        values = hexgrid(i[0],i[1],r)
        results.append(values)
    return results

def hexgrid_cells(cluster_size=7):
    assert cluster_size in (1, 7, 19), "Only 1, 7 and 19 cell clusters are supported"
    if cluster_size == 1:
        return [(0, 0)]
    elif cluster_size == 7:
        return [(0, 0), (0, -1), (1, -1), (-1, -1), (-1, 0), (0, 1), (1, 0)]
    elif cluster_size == 19:
        cls = [(0,0)]
        for cell_x in range(-2, 3):
            for cell_y in range(-2, 3):
                if abs(cell_x) + abs(cell_y) > 3 or (cell_y == 2 and abs(cell_x) == 1) or (cell_x == 0 and cell_y == 0):
                    continue
                cls.append((cell_x, cell_y))
        return cls


def hexgrid_box(size = 5):
    if size % 2 == 0:
        warn('Box hexgrid produces best results when size is odd')
    r = range(int(-(size // 2)), int((size + 1) // 2))
    return list(itertools.product(r, r))


if __name__ == "__main__":
    pos = ((-1, 1), (1, 1), (-2, 0), (2, 0), (-1, -2), (1, -2))
    print(hexgrid(0, 0, 0, 1))
    for p in pos:
        print(hexgrid(p[0], p[1], 0, 1))







