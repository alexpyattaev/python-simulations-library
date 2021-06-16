# This file has been modified from its original version.
#
# Original version is
# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Copyright (c) 2015-today, Alexander Pyattaev
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#


"""Homogeneous Transformation Matrices and Quaternions.

A library for calculating 4x4 matrices for translating, rotating, reflecting,
scaling, shearing, projecting, orthogonalizing, and superimposing arrays of
3D homogeneous coordinates as well as for converting between rotation matrices,
Euler angles, and quaternions. Also includes an Arcball control object and
functions to decompose transformation matrices.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_
   Dr. Alexander Pyattaev


Notes
-----
The API is not stable yet and is expected to change between revisions.

This Python code is not optimized for speed. Refer to the transformations.c
module for a faster implementation of some functions.

Documentation in HTML format can be generated with epydoc.

Matrices (M) can be inverted using np.linalg.inv(M), be concatenated using
np.dot(M0, M1), or transform homogeneous coordinate arrays (v) using
np.dot(M, v) for shape (4, \*) column vectors, respectively
np.dot(v, M.T) for shape (\*, 4) row vectors ("array of points").

This module follows the "column vectors on the right" and "row major storage"
(C contiguous) conventions. The translation components are in the right column
of the transformation matrix, i.e. M[:3, 3].
The transpose of the transformation matrices may have to be used to interface
with other graphics systems, e.g. with OpenGL's glMultMatrixd(). See also [16].

Calculations are carried out with np.float64 precision.

Vector, point, quaternion, and matrix function arguments are expected to be
"array like", i.e. tuple, list, or numpy arrays.

Return types are numpy arrays unless specified otherwise.

Angles are in radians unless specified otherwise.

Quaternions w+ix+jy+kz are represented as [w, x, y, z].

A triple of Euler angles can be applied/interpreted in 24 ways, which can
be specified using a 4 character string or encoded 4-tuple:

  *Axes 4-string*: e.g. 'sxyz' or 'ryxy'

  - first character : rotations are applied to 's'tatic or 'r'otating frame
  - remaining characters : successive rotation axis 'x', 'y', or 'z'

  *Axes 4-tuple*: e.g. (0, 0, 0, 0) or (1, 1, 1, 1)

  - inner axis: code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
  - parity : even (0) if inner axis 'x' is followed by 'y', 'y' is followed
    by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
  - repetition : first and last axis are same (1) or different (0).
  - frame : rotations are applied to static (0) or rotating (1) frame.

Other Python packages and modules for 3D transformations and quaternions:

* `Transforms3d <https://pypi.python.org/pypi/transforms3d>`_
   includes most code of this module.
* `Blender.mathutils <http://www.blender.org/api/blender_python_api>`_
* `numpy-dtypes <https://github.com/numpy/numpy-dtypes>`_

References
----------
(1)  Matrices and transformations. Ronald Goldman.
     In "Graphics Gems I", pp 472-475. Morgan Kaufmann, 1990.
(2)  More matrices and transformations: shear and pseudo-perspective.
     Ronald Goldman. In "Graphics Gems II", pp 320-323. Morgan Kaufmann, 1991.
(3)  Decomposing a matrix into simple transformations. Spencer Thomas.
     In "Graphics Gems II", pp 320-323. Morgan Kaufmann, 1991.
(4)  Recovering the data from the transformation matrix. Ronald Goldman.
     In "Graphics Gems II", pp 324-331. Morgan Kaufmann, 1991.
(5)  Euler angle conversion. Ken Shoemake.
     In "Graphics Gems IV", pp 222-229. Morgan Kaufmann, 1994.
(6)  Arcball rotation control. Ken Shoemake.
     In "Graphics Gems IV", pp 175-192. Morgan Kaufmann, 1994.
(7)  Representing attitude: Euler angles, unit quaternions, and rotation
     vectors. James Diebel. 2006.
(8)  A discussion of the solution for the best rotation to relate two sets
     of vectors. W Kabsch. Acta Cryst. 1978. A34, 827-828.
(9)  Closed-form solution of absolute orientation using unit quaternions.
     BKP Horn. J Opt Soc Am A. 1987. 4(4):629-642.
(10) Quaternions. Ken Shoemake.
     http://www.sfu.ca/~jwa3/cmpt461/files/quatut.pdf
(11) From quaternion to matrix and back. JMP van Waveren. 2005.
     http://www.intel.com/cd/ids/developer/asmo-na/eng/293748.htm
(12) Uniform random rotations. Ken Shoemake.
     In "Graphics Gems III", pp 124-132. Morgan Kaufmann, 1992.
(13) Quaternion in molecular modeling. CFF Karney.
     J Mol Graph Mod, 25(5):595-604
(14) New method for extracting the quaternion from a rotation matrix.
     Itzhack Y Bar-Itzhack, J Guid Contr Dynam. 2000. 23(6): 1085-1087.
(15) Multiple View Geometry in Computer Vision. Hartley and Zissermann.
     Cambridge University Press; 2nd Ed. 2004. Chapter 4, Algorithm 4.7, p 130.
(16) Column Vectors vs. Row Vectors.
     http://steve.hollasch.net/cgindex/math/matrix/column-vec.html

Examples
--------


>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>> S = scale_matrix(1.23, _origin)
>>> T = translation_matrix(vector(1, 2, 3))
>>> Z = shear_matrix(beta, xaxis, _origin, zaxis)
>>> R = random_rotation_matrix()
>>> M = concatenate_matrices(T, R, Z, S)
>>> scale, shear, angles, trans, persp = decompose_matrix(M)
>>> np.allclose(scale, 1.23)
True
>>> np.allclose(trans, [1, 2, 3])
True
>>> np.allclose(shear, [0, tan(beta), 0])
True
>>> is_same_transform(R, euler_matrix(axes='sxyz', *angles))
True
>>> M1 = compose_matrix(scale, shear, angles, trans, persp)
>>> is_same_transform(M, M1)
True
>>> v0, v1 = np.random.rand(3), np.random.rand(3)
>>> M = rotation_matrix(angle_between_vectors(v0, v1), np.cross(v0, v1))
>>> v2 = np.dot(v0, M[:3,:3].T)
>>> np.allclose(unit_vector(v1), unit_vector(v2))
True

>>> R = rotation_matrix(0.123, vector(1, 2, 3))
>>> q = quaternion_from_matrix(R)
>>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
True


"""
from typing import Tuple
from math import sqrt, sin, cos, asin, atan2, atan, tan
from math import pi
import numpy as np

from lib.transformations.euler_angles import euler_matrix
from lib.transformations.quaternions import quaternion_to_transform_matrix, quaternion_to_transformation_matrix,\
    random_quaternion, quaternion_from_matrix
from lib.transformations.transform_tools import is_same_transform, concatenate_matrices

from lib.stuff import EPS
from lib.numba_opt import jit_hardcore, double, complex128
from lib.vectors import vector, norm, unit_vector, angle_between_vectors, origin as _origin, xaxis, zaxis

__docformat__ = 'restructuredtext en'


@jit_hardcore
def translation_matrix(direction) -> np.ndarray:
    """Return matrix to translate by direction vector.

    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4, double)
    M[:3, 3] = direction[:3]
    return M


@jit_hardcore
def translation_from_matrix(matrix) -> np.ndarray:
    """Return translation vector from translation matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> np.allclose(v0, v1)
    True

    """
    return matrix[:3, 3].copy()


@jit_hardcore
def reflection_matrix(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return matrix to mirror at plane defined by point and normal vector.

    >>> v0 = np.random.random(4) - 0.5
    >>> v0[3] = 1.
    >>> v1 = np.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> np.allclose(2, np.trace(R))
    True
    >>> np.allclose(v0, np.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> np.allclose(v2, np.dot(R, v3))
    True

    """
    normal = unit_vector(normal[:3])
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return M


@jit_hardcore
def reflection_from_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = np.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> pt, normal_vector = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(pt, normal_vector)
    >>> is_same_transform(M0, M1)
    True

    """
    M = matrix
    # normal: unit eigenvector corresponding to eigenvalue -1
    w, V = np.linalg.eig(M[:3, :3])
    i = np.where(np.abs(np.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = np.real(V[:, i[0]]).flatten()
    # point: any unit eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(np.abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).flatten()
    point /= point[3]
    return point[0:3], normal


@jit_hardcore
def cross_vector_to_rotation(a):
    """
    Convert vector a into a rotation matrix R which would provide:
    R @ v = a x v

    :param a: vector a
    :return: rotation matrix
    >>> a = vector(0,0,1)
    >>> M = cross_vector_to_rotation(a)
    >>> M
    array([[ 0., -1.,  0.],
           [ 1.,  0., -0.],
           [-0.,  0.,  0.]])
    >>> v = vector(1,0,0)
    >>> np.allclose(M @ v, np.cross(a, v))
    True
    """
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


@jit_hardcore
def rotation_matrix(angle: float, direction: np.ndarray,
                    point: np.ndarray = _origin) -> np.ndarray:
    """Return matrix to rotate about axis defined by point and direction.
        :param angle: the angle for rotation
        :param direction: the direction vector of the rotation
        :param point: the reference point of rotation. By default origin is used.
        :returns: the resulting rotation matrix (4x4)
    
        >>> R = rotation_matrix(pi/2, vector(0, 0, 1), vector(1, 0, 0))
        >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
        True
        >>> angle = (np.random.random() - 0.5) * (2*pi)
        >>> direc = np.random.random(3) - 0.5
        >>> point = np.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*pi, direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> I = np.identity(4, np.float64)
        >>> np.allclose(I, rotation_matrix(pi*2, direc))
        True
        >>> np.allclose(2, np.trace(rotation_matrix(pi/2,
        ...                                               direc, point)))
        True
    
        """
    sina: float = sin(angle)
    cosa: float = cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag(np.array((cosa, cosa, cosa), dtype=double))
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if np.any(point != _origin):
        # rotation not around origin
        M[:3, 3] = point - np.dot(R, point)
    return M


@jit_hardcore
def rotation_from_matrix(matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return rotation angle and axis from rotation matrix.
    :returns: angle, direction, point
    >>> angle = (np.random.random() - 0.5) * (2*pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    RR33 = matrix[:3, :3]
    R = matrix.astype(complex128)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(R33.T)
    i = np.where(np.abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).flatten()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(R)
    i = np.where(np.abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).flatten()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(RR33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (RR33[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (RR33[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (RR33[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = atan2(sina, cosa)
    return angle, direction, point[0:3]


@jit_hardcore
def scale_matrix(factor: float, origin: np.ndarray = _origin, direction: np.ndarray = _origin):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (np.random.rand(4, 5) - 0.5) * 20
    >>> v[3] = 1
    >>> S = scale_matrix(-1.234)
    >>> np.allclose(np.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = np.random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if (direction == _origin).all():
        # uniform scaling
        M = np.diag(np.array((factor, factor, factor, 1.0), dtype=np.float64))
        if np.any(origin != _origin):
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if (origin == _origin).all():
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M


@jit_hardcore
def scale_from_matrix(matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return scaling factor, origin and direction from scaling matrix.

    >>> factor = np.random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """

    M33 = matrix[:3, :3]
    factor: float = np.trace(M33) - 2.0

    # direction: unit eigenvector corresponding to eigenvalue factor
    w, V = np.linalg.eig(M33)
    q = np.where(np.abs(np.real(w) - factor) < 1e-8)[0]
    if len(q):
        i = q[0]
        direction = np.real(V[:, i]).flatten()[0:3]
        direction /= norm(direction)
    else:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        direction = _origin
    # origin: any eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(matrix)
    i = np.where(np.abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    origin = np.real(V[:, i[-1]]).flatten()
    origin = (origin / origin[3])[0:3]
    return factor, origin, direction



def projection_matrix(point: np.ndarray, normal: np.ndarray, direction: np.ndarray = None,
                      perspective: np.ndarray = None, pseudo: bool = False) -> np.ndarray:
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = projection_matrix(vector(0, 0, 0), vector(1, 0, 0))
    >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
    True
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, direction=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, np.dot(P0, P3))
    True
    >>> P = projection_matrix(vector(3, 0, 0), vector(1, 1, 0), vector(1, 0, 0))
    >>> v0 = (np.random.rand(4, 5) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(P, v0)
    >>> np.allclose(v1[1], v0[1])
    True
    >>> np.allclose(v1[0], 3-v1[1])
    True

    """
    M = np.identity(4)
    point = np.copy(point[:3])
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.copy(perspective[:3])
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective - point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective + normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = np.copy(direction[:3])

        # noinspection PyTypeChecker
        scale: float = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1, eps=1e-7)
    True
    >>> P2 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P2)
    >>> P3 = projection_matrix(*result)
    >>> is_same_transform(P2, P3, eps=1e-6)
    True
    >>> P4 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P4, pseudo=False)
    >>> P5 = projection_matrix(*result)
    >>> is_same_transform(P4, P5, eps=1e-5)
    True
    >>> P6 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P6, pseudo=True)
    >>> P7 = projection_matrix(*result)
    >>> is_same_transform(P6, P7, eps=1e-7)
    True

    """
    M = matrix
    M33 = M[:3, :3]
    w, V = np.linalg.eig(M)
    i = np.where(np.abs(np.real(w) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = np.real(V[:, i[-1]]).flatten()
        point /= point[3]
        # direction: unit eigenvector corresponding to eigenvalue 0
        w, V = np.linalg.eig(M33)
        i = np.where(np.abs(np.real(w)) < 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector corresponding to eigenvalue 0")
        direction = np.real(V[:, i[0]]).flatten()
        direction /= norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        w, V = np.linalg.eig(M33.T)
        i = np.where(np.abs(np.real(w)) < 1e-8)[0]
        if len(i):
            # parallel projection
            normal = np.real(V[:, i[0]]).flatten()
            normal /= norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals direction vector
            return point, direction, None, None, False
    else:
        # perspective projection
        i = np.where(np.abs(np.real(w)) > 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector not corresponding to eigenvalue 0")
        point = np.real(V[:, i[-1]]).flatten()
        point /= point[3]
        normal = - M[3, :3]
        perspective = M[:3, 3] / np.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustum.

    The frustum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustum.

    If perspective is True the frustum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (divided by w coordinate).

    >>> frustum = np.random.rand(6)
    >>> frustum[1] += frustum[0]
    >>> frustum[3] += frustum[2]
    >>> frustum[5] += frustum[4]
    >>> M = clip_matrix(perspective=False, *frustum)
    >>> np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    array([-1., -1., -1.,  1.])
    >>> np.dot(M, [frustum[1], frustum[3], frustum[5], 1])
    array([1., 1., 1., 1.])
    >>> M = clip_matrix(perspective=True, *frustum)
    >>> v = np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = np.dot(M, [frustum[1], frustum[3], frustum[4], 1])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError("invalid frustum")
    if perspective:
        if near <= EPS:
            raise ValueError("invalid frustum: near <= 0")
        t = 2.0 * near
        M = [[t / (left - right), 0.0, (right + left) / (right - left), 0.0],
             [0.0, t / (bottom - top), (top + bottom) / (top - bottom), 0.0],
             [0.0, 0.0, (far + near) / (near - far), t * far / (far - near)],
             [0.0, 0.0, -1.0, 0.0]]
    else:
        M = [[2.0 / (right - left), 0.0, 0.0, (right + left) / (left - right)],
             [0.0, 2.0 / (top - bottom), 0.0, (top + bottom) / (bottom - top)],
             [0.0, 0.0, 2.0 / (far - near), (far + near) / (near - far)],
             [0.0, 0.0, 0.0, 1.0]]
    return np.array(M)


def shear_matrix(angle: float, direction: np.ndarray, point: np.ndarray, normal: np.ndarray):
    """Return matrix to shear by angle along direction vector on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> angle = (np.random.random() - 0.5) * 4*pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = shear_matrix(angle, direct, point, normal)
    >>> np.allclose(1, np.linalg.det(S))
    True

    """
    normal = unit_vector(normal[:3])
    direction = unit_vector(direction[:3])
    if abs(np.dot(normal, direction)) > 1e-6:
        raise ValueError("direction and normal vectors are not orthogonal")
    angle = tan(angle)
    M = np.identity(4)
    M[:3, :3] += angle * np.outer(direction, normal)
    M[:3, 3] = -angle * np.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """Return shear angle, direction and plane from shear matrix.

    >>> angle = (np.random.random() - 0.5) * 4*pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1, eps=1e-7)
    True

    """
    M = matrix
    M33 = M[:3, :3]
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    w, V = np.linalg.eig(M33)
    i = np.where(np.abs(np.real(w) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError("no two linear independent eigenvectors found")
    V = np.real(V[:, i]).squeeze().T
    lenorm = -1.0
    normal = None
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        w = norm(n)
        if w > lenorm:
            lenorm = w
            normal = n
    if normal is None:
        raise ValueError("Can not find normal vector!")
    normal /= lenorm
    # direction and angle
    direction = np.dot(M33 - np.identity(3), normal)
    angle = norm(direction)
    direction /= angle
    angle = atan(angle)
    # point: eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal


@jit_hardcore
def decompose_matrix(matrix: np.ndarray):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix(vector(1, 2, 3))
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> np.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> np.allclose(R0, R1)
    True

    """
    M = np.transpose(np.copy(matrix))

    if abs(M[3, 3]) < EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3,))
    shear = np.zeros(3)
    angles = np.zeros(3)

    if np.any(np.abs(M[:3, 3]) > EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = norm(row[2])
    row[2] /= scale[2]
    shear[1] /= scale[2]
    shear[2] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    angles[1] = asin(-row[0, 2])
    if cos(angles[1]):
        angles[0] = atan2(row[1, 2], row[2, 2])
        angles[2] = atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = atan2(row[1, 0], row[1, 1])
        angles[0] = atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """Return transformation matrix from sequence of transformations.

    This is the inverse of the decompose_matrix function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    >>> scale = np.random.random(3) - 0.5
    >>> shear = np.random.random(3) - 0.5
    >>> angles = (np.random.random(3) - 0.5) * (2*pi)
    >>> trans = np.random.random(3) - 0.5
    >>> persp = np.random.random(4) - 0.5
    >>> M0 = compose_matrix(scale, shear, angles, trans, persp)
    >>> result = decompose_matrix(M0)
    >>> M1 = compose_matrix(*result)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.identity(4)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]
        M = np.dot(M, P)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]
        M = np.dot(M, T)
    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = np.dot(M, R)
    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)
    M /= M[3, 3]
    return M


def orthogonalization_matrix(lengths, angles):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in degrees.

    The de-orthogonalization matrix is the inverse.

    >>> O = orthogonalization_matrix([10, 10, 10], [90, 90, 90])
    >>> np.allclose(O[:3, :3], np.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> np.allclose(np.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = np.radians(angles)
    sina, sinb, _ = np.sin(angles)
    cosa, cosb, cosg = np.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return np.array([
        [a * sinb * sqrt(1.0 - co * co), 0.0, 0.0, 0.0],
        [-a * sinb * co, b * sina, 0.0, 0.0],
        [a * cosb, b * cosa, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[1.45489948e-01, 6.24811295e-04, 6.75500083e+02],
           [4.84502149e-04, 1.40938160e-01, 5.32497108e+01],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> R = random_rotation_matrix()
    >>> S = scale_matrix(np.random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> np.allclose(v1, np.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx + yy + zz, 0.0, 0.0, 0.0],
             [yz - zy, xx - yy - zz, 0.0, 0.0],
             [zx - xz, xy + yx, yy - xx - zz, 0.0],
             [xy - yx, zx + xz, yz + zy, zz - xx - yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_to_transform_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    """Return matrix to transform given 3D point set into second point set.

    v0 and v1 are shape (3, \*) or (4, \*) arrays of at least 3 points.

    The parameters scale and usesvd are explained in the more general
    affine_matrix_from_points function.

    The returned matrix is a similarity or Euclidean transformation matrix.
    This function has a fast C implementation in transformations.c.

    >>> v0 = np.random.rand(3, 10)
    >>> M = superimposition_matrix(v0, v0)
    >>> np.allclose(M, np.identity(4))
    True
    >>> R = random_rotation_matrix()
    >>> v0 = [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]
    >>> v1 = np.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> S = scale_matrix(np.random.random())
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> M = concatenate_matrices(T, R, S)
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-9, 300).reshape(3, -1)
    >>> M = superimposition_matrix(v0, v1, scale=True)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> v = np.empty((4, 100, 3))
    >>> v[:, :, 0] = v0
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> np.allclose(v1, np.dot(M, v[:, :, 0]))
    True

    """
    v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False,
                                     scale=scale, usesvd=usesvd)


@jit_hardcore
def random_rotation_matrix():
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_to_transform_matrix(q))
    True

    """

    return quaternion_to_transformation_matrix(random_quaternion())


inverse_matrix = np.linalg.inv

if __name__ == "__main__":
    import doctest

    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
