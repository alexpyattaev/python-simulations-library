import numpy as np


def sphere_line_intersection(line: np.array, sphere: np.array) -> np.array:
    """Returns the two points which are the intersection of the given line and sphere.
    :param line: [x0 y0 z0  dx dy dz]
    :param sphere: [xc yc zc  R]
    :returns: intersection points ([x1 y1 z1], [x2 y2 z2])
    """

    tol = 1e-14

    # difference between centers
    dc = line[0:2] - sphere[0:2]

    # equation coefficients
    a = np.sum(line[3:5] * line[3:5])
    b = 2 * np.sum(dc * line[3:5])
    c = np.sum(dc ** 2) - sphere[3] * sphere[3]

    # solve equation
    delta = b * b - 4 * a * c

    if delta > tol:
        # delta positive: find two roots of second order equation
        u1 = (-b - np.sqrt(delta)) / 2 / a
        u2 = (-b + np.sqrt(delta)) / 2 / a

        if (u1 > 0) and (u2 > 0):
            # convert into 3D coordinate
            return (line[0:2] + u1 * line[3:5], line[0:2] + u2 * line[3:5])
        else:
            raise ValueError("No intersection")
    else:
        # delta negative: no solution
        raise ValueError("No intersection")


if __name__ == "__main__":
    l = np.array([0, 0, 0, 10, 0, 0])
    sphere = np.array([5, 0, 0, 3])
    print(sphere_line_intersection(l, sphere))
