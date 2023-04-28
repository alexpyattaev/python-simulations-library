import itertools
import pickle
from lib.numba_opt import jit_hardcore
import numpy as np
from lib.bresenham_algorithms.bresenham_algo_2D import Bresenham_circle

# Bresenham sphere
from lib.plots_stuff import axisEqual3D


def Bresenham_sphere(x0:int, y0:int, z0:int, radius:int, reflection=True):
    if radius < 3:
        raise ValueError('This does not work well with small radii')
    #todo = {(z, r) for z,r in Bresenham_circle(0, 0, radius, quarters=((-1, -1),(-1, 1)))}
    todo = {(z, r) for z, r in Bresenham_circle(0, 0, radius)}
    todo = sorted(todo)
    todo.append((radius, 0))
    #print(todo)

    for z, r in todo:
        #print(f'3d doing z={z}, r={r}')

        #for x, y in Bresenham_circle(x0, y0, r, quarters=((-1, -1), (-1, 1))):
        for x, y in Bresenham_circle(x0, y0, r):
            yield (x, y, z0 + z)
            #yield (x, y, z)
            # Flag to activate the building of the second part of the sphere
            if reflection is True and z != 0:
                #yield (x, y, z)
                yield ( x,  y, z0 - z)


# Bresenham rays generator
@jit_hardcore
def Bresenham3D(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while round(x1) != round(x2):
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            yield x1, y1, z1

            # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while round(y1) != round(y2):
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            yield x1, y1, z1

            # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while round(z1) != round(z2):
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            yield x1, y1, z1


# print('WTFFF')
# for p in sphere(10,10,10, 5):
#     print(p)
# print('WTFFFF')


def test_line():

    S = 512
    arr = np.zeros([S, S, S], dtype=bool)
    v1 = np.array([6, 0, 0])
    v2 = np.array([128, 0, 30])
    for p in Bresenham3D(*v1, *v2):
        arr[p[0],p[1],p[2]] = 1

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y, Z = filter_pointcloud(arr, BS=4)
    X, Y, Z = np.nonzero(arr)
    ax.scatter(xs=X, ys=Y, zs=Z, marker=".")
    axisEqual3D(ax)
    plt.show()


def Bresenham_Triangle(v1,v2,v3):
    edge1 = []
    edge2 = []

    for p in Bresenham3D(0, 0, 0, *(v2 - v1)):
        edge1.append(p)

    for p in Bresenham3D(*v1, *v3):
        edge2.append(p)

    lendec = len(edge1) / len(edge2)
    L = len(edge1)
    for q in edge2:
        for p in edge1[1:int(L)]:
            yield q[0] + p[0], q[1] + p[1], q[2] + p[2]
        L -= lendec


def Bresenham_Rectangle(v1,v2,v3):
    edge1 = []
    edge2 = []

    for p in Bresenham3D(0, 0, 0, *(v2 - v1)):
        edge1.append(p)

    for p in Bresenham3D(*v1, *v3):
        edge2.append(p)

    for q in edge2:
        for p in edge1:
            yield q[0] + p[0], q[1] + p[1], q[2] + p[2]


def test_triangle():
    S = 512
    arr = np.zeros([S, S, S], dtype=bool)
    v1 = np.array([6, 0, 0])
    v2 = np.array([128, 0, 80])
    v3 = np.array([300, 45, 0])

    for p in Bresenham_Triangle(v1,v2,v3):
        arr[p[0],p[1],p[2]] = 1

    X, Y, Z = filter_pointcloud(arr, BS=8)
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X = X/20
    # Y = Y/20
    # Z = Z/20
    # #X, Y, Z = np.nonzero(arr)
    # ax.scatter(xs=X, ys=Y, zs=Z, marker=".")
    # axisEqual3D(ax)
    # plt.show()
    return(len(X))


def test_rectangle():
    S = 512
    arr = np.zeros([S, S, S], dtype=bool)
    v1 = np.array([0, 0, 0])
    v3 = np.array([200, 0, 0])
    v2 = np.array([0, 50, 80])

    for p in Bresenham_Rectangle(v1,v2,v3):
        arr[p[0],p[1],p[2]] = 1
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = filter_pointcloud(arr, BS=8)
    #X, Y, Z = np.nonzero(arr)
    ax.scatter(xs=X, ys=Y, zs=Z, marker=".")
    axisEqual3D(ax)
    plt.show()

def filter_pointcloud(arr, BS=4):
    X, Y, Z = [], [], []
    S = len(arr)
    for x, y, z in itertools.product(range(S // BS), range(S // BS), range(S // BS)):
        block = arr[x * BS:(x + 1) * BS, y * BS:(y + 1) * BS, z * BS:(z + 1) * BS]
        a, b, c = np.nonzero(block)
        if len(a):
            X.append(np.mean(a) / BS + x)
            Y.append(np.mean(b) / BS + y)
            Z.append(np.mean(c) / BS + z)
    return X, Y, Z


def test_sphere():
    S = 512
    arr = np.zeros([S, S, S], dtype=bool)

    for q in Bresenham_sphere(S//2, S//2, S//2, int(S*0.4)):
        arr[q[0], q[1], q[2]] = 1

    X, Y, Z = filter_pointcloud(arr, BS=4)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    k = (max(X)-min(X))/2 + min(X)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    X = X - k
    Y = Y - k
    Z = Z - k
    pickle_out = open("C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_UAV/article 2/bresenham_sphere.pickle","wb")
    pickle.dump([X,Y,Z], pickle_out)
    pickle_out.close()

    #X, Y, Z = np.nonzero(arr)
    ax.scatter(xs=X, ys=Y, zs=Z, marker=".")
    #ax.plot(X, Y, zs=Z, marker=".", linestyle=None)
    axisEqual3D(ax)
    plt.show()
#
# time1 = datetime.datetime.now()
# len_points = test_triangle()
# time2 = datetime.datetime.now()
# print("Number of points (first approach): ",len_points)
# print("Time (first approach): ",time2 - time1)
# #test_rectangle()