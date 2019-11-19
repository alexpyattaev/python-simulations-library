
import sys
import numpy as np
from numpy import array

from numpy import concatenate
from numpy import sum
from numpy.linalg.linalg import pinv
from numpy.linalg.linalg import LinAlgError
from scipy.spatial.distance import euclidean


def transpose_1D(M):
    """numpy is silly and won't let you transpose a N x 1 ndarray to a 1 x N
    ndarray.
    :param M: input array, assumed to be 1-D
    :returns: M transposed
    """
    return M.reshape(len(M), 1)


def multilaterate(anchor_positions, distances):
    """Multilaterate based on anchor positions and distances
        :param anchor_positions: Nx3 array of 3d coordinates
        :param distances: N distances to anchors
        :returns: 3-vector of estimated position
        """
    N = anchor_positions.shape[0]
    A = np.vstack([np.ones(N), -2 * anchor_positions[:, 0],
                   -2 * anchor_positions[:, 1], -2 * anchor_positions[:, 2]]).T
    B = distances ** 2 - anchor_positions[:, 0] ** 2 - anchor_positions[:, 1] ** 2 - anchor_positions[:, 2] ** 2
    #print("A", A)
    #print("B", B)
    # TODO: use np.linalg.pinv()?
    X = np.dot(A.T, A)
    #print(X)
    xp = np.dot(np.dot(np.linalg.inv(X), A.T), B)

    return xp[1:]


if __name__ == '__main__':
    ANCHOR_POS = array([
        [0, 0, 0],
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
        [10, 0, 10],
    ])
    stat_errs= []
    dists_cm = [10,50,100]
    variances = np.array([0.03, 0.15, 0.6])
    stat_nanch = []
    Q = 200
    S = 20.0
    for N_ANCH in range(4, 20):
        stat_nanch.append(N_ANCH)
        mean_err = np.zeros_like(variances)

        for mp in range(Q):
            ANCHOR_POS = np.random.randint(0, 5, [N_ANCH, 3]) * S / 5 + np.random.uniform(-0.5, 0.5, [N_ANCH, 3])

            MY_POS = np.random.uniform(0, S, 3)

            DISTANCES_METERS = array([euclidean(p, MY_POS) for p in ANCHOR_POS])
            #Use 0.3 variance to get 50cm mean measurement error in raw data
            # Use 0.05 variance to get 10 cm mean measurement error in raw data
            for i, v in enumerate(variances):
                ERR = np.random.normal(0, v, len(ANCHOR_POS))
                res = multilaterate(ANCHOR_POS, DISTANCES_METERS + ERR)
                mean_err[i] += euclidean(res, MY_POS)
        print(mean_err)
        stat_errs.append(mean_err / Q)

    import matplotlib.pyplot as plt
    stat_errs = np.array(stat_errs).T
    print(stat_errs)
    plt.figure()
    for d, v, e in zip(dists_cm, variances, stat_errs):

        plt.plot(stat_nanch, e, label=f"{d} cm ranging error (variance {v})")
    plt.hlines(0.1, min(stat_nanch)+1, max(stat_nanch)-1, label="10cm error target", linewidth=0.3)
    plt.xlabel("Number of anchors")
    plt.ylabel("Positioning error, m")
    
    plt.legend()
    plt.grid()
    plt.show(block=True)
