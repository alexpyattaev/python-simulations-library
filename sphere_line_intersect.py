from typing import List


from numpy import sum, sqrt, array, ndarray
from lib import jit_hardcore, jit


def line_sphere_intersection(line_point1: ndarray, line_point2: ndarray,
                             sphere_center: ndarray, sphere_radius: float,
                             is_ray: bool = False) -> List[ndarray]:
    """Returns the two points which are the intersection of the given line and sphere.
    :param line_point1: [x0 y0 z0]
    :param line_point2: [x1 y1 z1]
    :param sphere_center: [xc yc zc]
    :param sphere_radius: r
    :param is_ray: True if only ray intersection is needed
    :returns: List of intersection points (can be empty)
    """

    cx, cy, cz = sphere_center
    px, py, pz = line_point1
    vx = line_point2[0] - px
    vy = line_point2[1] - py
    vz = line_point2[2] - pz

    A = vx * vx + vy * vy + vz * vz
    B = 2.0 * (px * vx + py * vy + pz * vz - vx * cx - vy * cy - vz * cz)
    C = px * px - 2 * px * cx + cx * cx + py * py - 2 * py * cy + cy * cy + pz * pz - \
        2 * pz * cz + cz * cz - sphere_radius * sphere_radius

    D = B * B - 4 * A * C

    if D < 0:
        return []

    t1 = (- B - sqrt(D)) / (2.0 * A)
    solution1 = line_point1 * (1 - t1) + line_point2 * t1

    if D == 0:
        if is_ray and t1 < 0:
            return []
        return [solution1, ]

    t2 = (- B + sqrt(D)) / (2.0 * A)
    solution2 = line_point1 * (1 - t2) + line_point2 * t2

    if is_ray:
        if t1 > 0 and t2 > 0:
            return [solution1, solution2]
        elif (t1 > 0) and (t2 < 0):
            return [solution1, ]
        elif (t1 < 0) and (t2 > 0):
            return [solution2, ]
        else:
            return []

    return [solution1, solution2]


def ray_sphere_intersection(origin: ndarray, direction: ndarray, sphere_pos: ndarray, sphere_R: float) -> List[ndarray]:
    """Returns the two points which are the intersection of the given line and sphere.
    :param origin: [x0 y0 z0]
    :param direction: [dx0 dy0 dz0]
    :param sphere_pos: [xc yc zc]
    :param sphere_R: r
    :returns: intersection points [[x1 y1 z1], [x2 y2 z2]] or empty list
    """

    tol = 1e-14

    # difference between centers
    dc = origin - sphere_pos

    # equation coefficients
    a = sum(direction * direction)
    b = 2 * sum(dc * direction)
    c = sum(dc ** 2) - sphere_R**2

    # solve equation
    delta = b * b - 4 * a * c

    if delta > tol:
        # delta positive: find two roots of second order equation
        u1 = (-b - sqrt(delta)) / 2 / a
        u2 = (-b + sqrt(delta)) / 2 / a

        if (u1 > 0) and (u2 > 0):
            # convert into 3D coordinate
            return [origin + u1 * direction, origin + u2 * direction]
        elif u1 > 0:
            return [origin + u1 * direction]
        elif u2 > 0:
            return [origin + u2 * direction]
        else:
            return []
    else:
        # delta negative: no solution
        return []


