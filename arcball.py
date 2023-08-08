import math

import numpy as np

from lib.stuff import EPS
from lib.transformations.quaternions import quaternion_from_matrix, quaternion_slerp, quaternion_multiply, \
    quaternion_from_elements, quaternion_to_transform_matrix
from lib.vectors import norm, unit_vector, vector


class Arcball(object):
    """Virtual Trackball Control.

    >>> ball = Arcball()
    >>> ball = Arcball(initial=np.ident(4))
    >>> ball.place([320, 320], 320)
    >>> ball.down([500, 250])
    >>> ball.drag([475, 275])
    >>> R = ball.matrix()
    >>> np.allclose(np.sum(R), 3.90583455)
    True
    >>> ball = Arcball(initial=[1, 0, 0, 0])
    >>> ball.place([320, 320], 320)
    >>> ball.setaxes([1, 1, 0], [-1, 1, 0])
    >>> ball.constrain = True
    >>> ball.down([400, 200])
    >>> ball.drag([200, 400])
    >>> R = ball.matrix()
    >>> np.allclose(np.sum(R), 0.2055924)
    True
    >>> ball.next()

    """

    def __init__(self, initial=None):
        """Initialize virtual trackball control.

        initial : quaternion or rotation matrix

        """
        self._axis = None
        self._axes = None
        self._radius = 1.0
        self._center = [0.0, 0.0]
        self._vdown = np.array([0.0, 0.0, 1.0])
        self._constrain = False
        if initial is None:
            self._qdown = quaternion_from_elements(1.0, 0.0, 0.0, 0.0)
        else:
            initial = np.array(initial, dtype=float64)
            if initial.shape == (4, 4):
                self._qdown = quaternion_from_matrix(initial)
            elif initial.shape == (4,):
                initial /= norm(initial)
                self._qdown = initial
            else:
                raise ValueError("initial not a quaternion or matrix")
        self._qnow = self._qpre = self._qdown

    def place(self, center, radius):
        """Place Arcball, e.g. when window size changes.

        center : sequence[2]
            Window coordinates of trackball center.
        radius : float
            Radius of trackball in window coordinates.

        """
        self._radius = float(radius)
        self._center[0] = center[0]
        self._center[1] = center[1]

    def setaxes(self, *axes):
        """Set axes to constrain rotations."""
        if axes is None:
            self._axes = None
        else:
            self._axes = [unit_vector(axis) for axis in axes]

    @property
    def constrain(self):
        """Return state of constrain to axis mode."""
        return self._constrain

    @constrain.setter
    def constrain(self, value):
        """Set state of constrain to axis mode."""
        self._constrain = bool(value)

    def down(self, point):
        """Set initial cursor window coordinates and pick constrain-axis."""
        self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
        self._qdown = self._qpre = self._qnow
        if self._constrain and self._axes is not None:
            self._axis = arcball_nearest_axis(self._vdown, self._axes)
            self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
        else:
            self._axis = None

    def drag(self, point):
        """Update current cursor window coordinates."""
        vnow = arcball_map_to_sphere(point, self._center, self._radius)
        if self._axis is not None:
            vnow = arcball_constrain_to_axis(vnow, self._axis)
        self._qpre = self._qnow
        t = np.cross(self._vdown, vnow)
        if np.dot(t, t) < EPS:
            self._qnow = self._qdown
        else:
            q = quaternion_from_elements(np.dot(self._vdown, vnow), t[0], t[1], t[2])
            self._qnow = quaternion_multiply(q, self._qdown)

    def next(self, acceleration=0.0):
        """Continue rotation in direction of last drag."""
        q = quaternion_slerp(self._qpre, self._qnow, 2.0 + acceleration, False)
        self._qpre, self._qnow = self._qnow, q

    def matrix(self):
        """Return homogeneous rotation matrix."""
        return quaternion_to_transform_matrix(self._qnow)


def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0 * v0 + v1 * v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return np.array([v0 / n, v1 / n, 0.0])
    else:
        return np.array([v0, v1, math.sqrt(1.0 - n)])


def arcball_constrain_to_axis(point, axis):
    """Return sphere point perpendicular to axis."""
    v = np.array(point, dtype=np.float64, copy=True)
    a = np.array(axis, dtype=np.float64, copy=True)
    v -= a * np.dot(a, v)  # on plane
    n = norm(v)
    if n > EPS:
        if v[2] < 0.0:
            np.negative(v, v)
        v /= n
        return v
    if a[2] == 1.0:
        return vector(1.0, 0.0, 0.0)
    return unit_vector(vector(-a[1], a[0], 0.0))


def arcball_nearest_axis(point, axes):
    """Return axis, which arc is nearest to point."""
    point = np.array(point, dtype=np.float64, copy=False)
    nearest = None
    mx = -1.0
    for axis in axes:
        t = np.dot(arcball_constrain_to_axis(point, axis), point)
        if t > mx:
            nearest = axis
            mx = t
    return nearest