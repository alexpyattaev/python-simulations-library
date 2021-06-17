from typing import List

import matplotlib.pyplot as plt
import numpy as np

from lib.plots_stuff import draw_point_labels
from lib.transformations.transformations import reflection_matrix, rotation_matrix, rotation_from_matrix
from lib.transformations.quaternions import quaternion_from_matrix, quaternion_to_transformation_matrix
from lib.vectors import norm


def rotation_to_match(target_vector, actual_vector):
    """
    Construct rotation matrix that would bring actual_vector to coincide with target_vector.

    Both input vectors must be normalized!!!!
    Works very reliably for all tested datasets

    :param target_vector: 3-vector of where we want to appear
    :param actual_vector: 3-vector of where we are
    :return: 3x3 rotation matrix
    """
    # TODO: rewrite with quaternions?
    print(f"actual = {actual_vector}, target={target_vector}")
    # Check if we even need to do anything =)
    if np.allclose(target_vector, actual_vector):
        return np.eye(3)

    # an off-plane vector is needed for rotations, lets make it using cross product
    zz = np.cross(actual_vector, target_vector)
    print("zz=", zz)

    # check for gimbal lock =)
    if np.allclose(zz, np.zeros_like(zz)):
        print("No cross product can be found!!!")
        # luckily, the only case when cross product is zero corresponds to reflection
        # the case of noop is taken care of previously
        rmd = reflection_matrix(np.zeros(3), actual_vector)[:3, :3]
    else:
        # normalize the off-plane vector
        zz = zz / norm(zz)
        # Create a 90-degree rotation matrix using off-plane vector to construct new Y axis
        rz = rotation_matrix(np.pi / 2, zz)[:3, :3]
        # construct the new "y" axis using rotation of actual_vector
        q1 = rz @ actual_vector
        # plot_line(ax,np.zeros(3), q1, label="q1", linestyle=":", color="black")
        rm0 = np.vstack((actual_vector, q1, zz)).T

        # construct the new "y" axis using rotation of target_vector
        q2 = rz @ target_vector
        # plot_line(ax,np.zeros(3), q2, label="q2", linestyle=":", color="green")
        rm1 = np.vstack((target_vector, q2, zz)).T

        # Difference between rotations will be what we actually need=)
        rmd = rm1 @ rm0.T
    return rmd


def interpolate_rotation_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """Interpolates between multiple rotation matrices as to produce some compromise.

    Uses quaternion linear interpolation (NLerp) internally. Does not work for large rotations
    :param matrices: list of 3x3 rotation matrices
    :returns: 3x3 rotation matrix
    """

    quats = [quaternion_from_matrix(rm) for rm in matrices]

    qsum = quatWAvgMarkley(np.array(quats), np.ones(len(quats)) / len(quats))

    return quaternion_to_transformation_matrix(qsum)


def quatWAvgMarkley(Q, weights: np.ndarray):
    '''
    Averaging Quaternions. Does not work if they are too different

    Arguments:
        Q(ndarray): an Mx4 ndarray of quaternions.
        weights(ndarray): an M elements array, a weight for each quaternion.
    '''

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4), dtype=float)
    wSum = weights.sum()
    print("Q", Q)
    for q, w_i in zip(Q, weights):
        A += w_i * np.outer(q, q)  # rank 1 update

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    print("A", A)
    print(np.linalg.eigh(A))
    return np.linalg.eigh(A)[1][:, -1]


def get_rotation_to_match_offsets(P: np.ndarray, Q: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Uses Kabsch algorithm to find a rotation and translation of P such that it matches Q in least squares sense.

    Details of algo: https: // en.wikipedia.org / wiki / Kabsch_algorithm
    Important note: the least squares sense does not always make sense for people!
    :param P: The desired locations, do not need to be centered
    :param Q: The existing locations to be matched
    :return: [R, offset], where R is the rotation matrix, and offset is the centroid of Q
    """
    # Remove the offsets to centroids
    cent_P = np.mean(P, axis=0)

    cent_Q = np.mean(Q, axis=0)

    P = P.copy() - cent_P
    Q = Q.copy() - cent_Q

    # Just fucking magick (tm)
    H = P.T @ Q
    U, S, V = np.linalg.svd(H)

    D = np.eye(3)
    D[2, 2] = np.linalg.det(V @ U.T)

    # Get rotation matrix
    R = V @ D @ U.T

    return R, cent_P, cent_Q


if __name__ == "__main__":
    A = np.array
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    P = A([[0, 0, 0],
           [10, 0, 0],
           [10, 10, 0],
           [0, 10, 0]], dtype=float)

    Q = A([[5, 0, 0],
           [6, 0, 0],
           [17, 8, 0],

           [2, 14, 5]], dtype=float)

    plt.plot(P[:, 0], P[:, 1], ":*r", zs=P[:, 2], label="Original P", linewidth=0.8)
    draw_point_labels(ax, P, color="red")
    plt.plot(Q[:, 0], Q[:, 1], ":*b", zs=Q[:, 2], label="Original Q", linewidth=0.8)
    draw_point_labels(ax, Q, color="blue")

    R, cent_P, cent_Q = get_rotation_to_match_offsets(P, Q)
    plt.plot([cent_P[0]], [cent_P[1]], "or", zs=[cent_P[2]], label="centroid P")
    P = P.copy() - cent_P
    plt.plot([cent_Q[0]], [cent_Q[1]], "ob", zs=[cent_Q[2]], label="centroid Q")

    R2 = np.eye(4)
    R2[:3, :3] = R
    ang, axis, pt, = rotation_from_matrix(R2)
    print(f"Angle {np.degrees(ang)} deg, axis {axis}, origin {pt}")

    # New points
    Q2 = (R @ P.T).T + cent_Q

    plt.plot(Q2[:, 0], Q2[:, 1], ":+g", zs=Q2[:, 2], label="new Q", linewidth=0.8)
    draw_point_labels(ax, Q2, color="green")
    plt.legend()
    plt.grid()
    # plt.axis('equal')
    plt.show()
