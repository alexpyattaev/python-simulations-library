"""
>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>>
>>> I = np.identity(4)
>>> Rx = rotation_matrix(alpha, xaxis)
>>> Ry = rotation_matrix(beta, yaxis)
>>> Rz = rotation_matrix(gamma, zaxis)
>>> R = concatenate_matrices([Rx, Ry, Rz])
>>> euler = euler_from_matrix(R, 'rxyz')
>>> np.allclose([alpha, beta, gamma], euler)
True
>>> Re = euler_matrix(alpha, beta, gamma, 'rxyz')
>>> is_same_transform(R, Re)
True
>>> al, be, ga = euler_from_matrix(Re, 'rxyz')
>>> is_same_transform(Re, euler_matrix(al, be, ga, 'rxyz'))
True
>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>>
>>> I = np.identity(4)
>>> Rx = rotation_matrix(alpha, xaxis)
>>> Ry = rotation_matrix(beta, yaxis)
>>> Rz = rotation_matrix(gamma, zaxis)
>>> R = concatenate_matrices([Rx, Ry, Rz])
>>> euler = euler_from_matrix(R, 'rxyz')
>>> np.allclose([alpha, beta, gamma], euler)
True

"""
import numpy as np

from lib.transformations.euler_angles import euler_from_matrix, euler_matrix
from lib.transformations.transform_tools import concatenate_matrices, is_same_transform
from lib.transformations.transformations import rotation_matrix
from lib.vectors import yaxis, xaxis, zaxis
