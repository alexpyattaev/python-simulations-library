import itertools
from typing import Tuple

import numpy as np
from lib.numba_opt import jit_hardcore

# Based on
# http://members.chello.at/easyfilter/bresenham.html


def Bresenham_AA(p1, p2) -> Tuple[Tuple[int, int], float]:
    """
    Iterate in all points that are touched by line from p1 to p2
    :param p1: start point, (x,y)
    :param p2: end point, (x,y)
    :return: tuple of coordinates (x,y) and color (float from 0 to 1)
    """
    x0, y0 = p1
    x1, y1 = p2
    dx = int(abs(x1 - x0))
    if x0 < x1:
        sx = 1
    else:
        sx = -1

    dy = int(abs(y1 - y0))
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    if dx + dy == 0:
        ed = 1
    else:
        ed = round(np.sqrt(dx * dx + dy * dy))

    while True:  # pixel loop
        yield (x0, y0), 1 - abs(err - dx + dy) / ed

        e2 = err
        x2 = x0

        if 2 * e2 >= -dx:  # /* x step */
            if x0 == x1:
                break
            if e2 + dy < ed:
                yield (x0, y0 + sy), 1 - (e2 + dy) / ed
            err -= dy
            x0 += sx

        if 2 * e2 <= dy:  # /* y step */
            if y0 == y1:
                break
            if dx - e2 < ed:
                yield (x2 + sx, y0), 1 - (dx - e2) / ed
            err += dx
            y0 += sy


ALL_QUARTERS = tuple(itertools.product([-1, 1], [-1, 1]))

@jit_hardcore
def Bresenham_circle(x0, y0, radius, quarters=ALL_QUARTERS):
    if radius == 0:
        yield x0, y0
        return
    elif radius == 1:  # FIXME misses points
        for q1, q2 in quarters:
            yield x0 + q1, y0 + q2
            yield x0 + q1, y0 + q2
        return
    f = 1 - radius
    ddf_x = 1
    ddf_y = -2 * radius
    x = 0
    y = radius
    for q1, q2 in quarters:
        if q1 * q2 > 0:
            yield x0 + x * q1, y0 + y * q2
        else:
            yield x0 + y * q1, y0 + x * q2
    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x
        for q1, q2 in quarters:
            yield x0 + x * q1, y0 + y * q2
            if x != y:
                yield x0 + y * q1, y0 + x * q2
