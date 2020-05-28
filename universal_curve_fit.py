import bezier
import numpy as np
from scipy.interpolate import interp1d
from typing import Iterable

__author__ = "Henrik Talarmo"


def fit_bezier_curve(x_points: np.ndarray, y_points: np.ndarray, derivatives: Iterable[float] = None,
                        resolution: int = 10, aggressiveness: float = 0.5) -> interp1d:
    """
    Creates a function for a curve defined by a set of points and the derivatives at those points. It uses bezier curves
    to define curves between the nodes and constructs a continuous curve based on those.

    :param x_points:
        1-dimensional array containing the x-coordinates of the nodes. Assumes that the nodes are in order in x-axis.
    :type x_points:
        numpy.ndarray
    :param y_points:
        1-dimensional array containing the y-coordinates of the nodes.
    :type y_points:
        numpy.ndarray
    :param derivatives:
        1-dimensional array containing all the derivatives of the nodes. The bezier curve will ensure that the resulting
        curve's derivative is 0 at these points. If set to None, it will assume derivative of 0 for all nodes.
    :type derivatives:
        Any iterable
    :param resolution:
        Defines the smoothness of the curve before interpolation.
    :type resolution:
        int
    :param aggressiveness:
        Defines how aggressively the bezier curves between nodes are defined. Must be between 0 and 0.5. If set to 0 the
        end result is effectively a linear interpolation between the points.
    :type aggressiveness:
        float
    :return:
        Returns an interp1d object that acts as a function that defines the given curve defined from x_points[0] to
        x_points[-1]
    :rtype:
        scipy.interpolate.interp1d object
    """
    assert 0 < aggressiveness <= 0.5, "Valid range for aggressiveness is 0..0.5"
    if derivatives is None:
        derivatives = np.zeros(shape=x_points.shape)
    assert len(derivatives) == len(x_points) == len(y_points), "All arrays must be the same length"

    # Create node lists containing all the points of the full curve
    curve_x_vals = np.array([x_points[0]])
    curve_y_vals = np.array([y_points[0]])
    for i in range(len(x_points)-1):
        # Find node points
        node1_x = x_points[i]
        node1_y = y_points[i]
        node1_d = derivatives[i]

        node2_x = x_points[i+1]
        node2_y = y_points[i+1]
        node2_d = derivatives[i+1]

        # Calculate x and y coordinates of the bezier nodes
        mid_x1 = node1_x + (node2_x - node1_x)*aggressiveness
        mid_x2 = node2_x - (node2_x - node1_x)*aggressiveness

        mid_y1 = node1_y + node1_d * (mid_x1 - node1_x)
        mid_y2 = node2_y + node2_d * (mid_x2 - node2_x)

        nodes_x = [node1_x, mid_x1, mid_x2, node2_x]
        nodes_y = [node1_y, mid_y1, mid_y2, node2_y]

        # Create bezier curve
        nodes = np.asfortranarray([nodes_x, nodes_y])
        curve = bezier.Curve(nodes, degree=3)

        # Calculate points along the bezier curve and add them to the total curve list
        xaxis = np.linspace(0, 1.0, resolution)

        ret = curve.evaluate_multi(xaxis)
        curve_x_vals = np.concatenate([curve_x_vals, ret[0][1:]], axis=0)
        curve_y_vals = np.concatenate([curve_y_vals, ret[1][1:]], axis=0)

    # interpolate curve and return it
    return interp1d(curve_x_vals, curve_y_vals, kind='linear',fill_value="extrapolate", bounds_error = False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x_points = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
    y_points = [0, 5, 25, 5, 20, 0, 15, -5, 0, 0]
    derivatives = [0, 0.3, 0, -0.1, -0.1, -0.07, -0.05, 0]
    f = fit_bezier_curve(x_points=np.array(x_points), y_points=np.array(y_points), resolution=50, aggressiveness=0.5)

    xnew = np.linspace(x_points[0], x_points[-1], num=100, endpoint=True)
    plt.plot(x_points, y_points, "*")
    plt.plot(xnew, f(xnew))
    plt.show()