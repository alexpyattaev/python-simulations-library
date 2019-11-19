from math import pi, trunc, sin, cos
import numpy as np
from .stuff import sign
_pi2 = 2 * pi
__author__ = 'Alex Pyattaev'


def pi_wrap(x):
    #Remove all full circles
    x = np.array(x, dtype=float)
    x %= _pi2
    #Ensure that nothing is overshooting pi
    x[x > pi] -= _pi2
    x[x <- pi] += _pi2
    return x


def ang_square(a):
    """Return a "squared" version of the angle, suitable for rectangular grids sometimes"""
    c=cos(a)
    s=sin(a)
    x=0
    y=0
    if abs(c)>0.1:
        x=sign(c)
    if abs(s)>0.1:
        y=sign(s)
    return x,y

# for i in np.arange(10):
#     print(np.degrees(i), np.degrees(pi_wrap(i)))


def sph2cart(R, th, phi):
    sin_th = np.sin(th)
    x = R * np.outer(np.cos(phi),sin_th)
    y = R * np.outer(np.sin(phi),sin_th)
    z = R * np.outer(np.ones_like(phi), np.cos(th))
    return x, y, z


def cart2sph1(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-15
    th = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, th, phi


