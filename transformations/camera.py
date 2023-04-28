"""
Basic examples to illustrate work with camera projections. Should explain most things naturally, everything else can
be found in http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#the-model-matrix
"""
import matplotlib.pyplot as plt
import numpy as np

from lib.numba_opt import jit_hardcore
from lib.plots_stuff import plot_line
from lib.transformations.euler_angles import euler_matrix
from lib.transformations.transformations import projection_matrix, project_by_matrix, translation_matrix, homogenous, \
    cartesian, compose_matrix, \
    string_decomposition, clip_matrix, scale_matrix
from lib.vectors import vector, vector_normalize


@jit_hardcore
def look_at(eyePosition3D, center3D, upVector3D):
    """
    Reimplementation of lookat from opengl. Very likely broken very much.
    """
    matrix = np.zeros(16)

    forward = vector_normalize(center3D - eyePosition3D)

    # Side = forward x up
    side = np.zeros(3)
    side[0] = (forward[1] * upVector3D[2]) - (forward[2] * upVector3D[1])
    side[1] = (forward[2] * upVector3D[0]) - (forward[0] * upVector3D[2])
    side[2] = (forward[0] * upVector3D[1]) - (forward[1] * upVector3D[0])
    side = vector_normalize(side)
    # Recompute up as: up = side x forward
    up = np.zeros(3)
    up[0] = (side[1] * forward[2]) - (side[2] * forward[1])
    up[1] = (side[2] * forward[0]) - (side[0] * forward[2])
    up[2] = (side[0] * forward[1]) - (side[1] * forward[0])
    #print(side, up, forward)
    matrix[0] = side[0]
    matrix[4] = side[1]
    matrix[8] = side[2]
    # matrix[12] = 0.0
    # --------------------
    matrix[1] = up[0]
    matrix[5] = up[1]
    matrix[9] = up[2]
    # matrix[13] = 0.0
    # --------------------
    matrix[2] = -forward[0]
    matrix[6] = -forward[1]
    matrix[10] = -forward[2]
    # matrix[14] = 0.0
    # --------------------
    # matrix[3] = matrix[7] = matrix[11] = 0.0
    matrix[15] = 1.0
    # --------------------
    res = np.reshape(matrix, (4, 4))
    tm = translation_matrix(-eyePosition3D)

    return res @ tm


def demo():
    xsize = 1
    ysize = 1
    #model_matrix = np.ident(4)
    model_matrix = translation_matrix(vector(1.0, 0, 0))@euler_matrix(ak=np.pi/4)@scale_matrix(0.8)

    view_matrix = compose_matrix(angles=[0.0, 0., np.pi], translate=vector(0.0, 0.0, -7))

    print("matrix view_matrix:", string_decomposition(view_matrix))
    # cam_pos = vector(0, 5, 10)
    # view_matrix = look_at(cam_pos, origin, vector(.0, .0, 1))
    # print("lookat view_matrix:", string_decomposition(view_matrix))


    #persp = vector(0.0, 0.0, 0.0)
    #camera_projection = projection_matrix(point=vector(0.0, 0.0, 1.0), normal=vector_normalize(vector(0.0, 0.0, 1.0)),
    #                                      perspective=persp)
    #print(f"{camera_projection=}")

    camera_projection = clip_matrix(-xsize, xsize, -ysize, ysize, 1.0, 10.0, perspective=True)
    print(f"{camera_projection=}")
    #exit()

    mesh1 = np.array([[-1, 0, 0], [0, 3, 0], [1, 0, 0], [-1, 0, 0]], dtype=float)
    mesh2 = np.array([[-1, 0, 2], [0, 3, 2], [1, 0, 2], [-1, 0, 2]], dtype=float)
    mesh3 = np.array([[-1, 0, 4], [0, 3, 4], [1, 0, 4], [-1, 0, 4]], dtype=float)
    meshes = [mesh1, mesh2, mesh3]


    colors = 'rgb'

    plt.figure(figsize=[xsize*5, ysize*5])
    plt.title("'Full-path' process doing transforms one by one. Inefficient but easy to understand.")
    ax = plt.gca()
    for mesh, color in zip(meshes, colors):
        print("=======")
        print("Transfer the points into view position")
        pts_model = [model_matrix @ homogenous(p) for p in mesh]
        print(f"{pts_model=}")
        print("Transfer the points into view position")
        pts_modelview = [view_matrix @ p for p in pts_model]
        print(f"{pts_modelview=}")
        print("Project the points into camera plane")
        pts_proj = [(camera_projection@p) for p in pts_modelview]
        print(f"{pts_proj=}")
        print("Drop into the 2D representation")
        pts_cart = [cartesian(p)[0:2] for p in pts_proj]
        print(f"{pts_cart=}")
        for i in range(len(pts_cart)-1):
            plot_line(ax, pts_cart[i][0:2], pts_cart[i+1][0:2], color=color)


    plt.xlim([-xsize, xsize])
    plt.ylim([-ysize, ysize])
    print("+++++++++++++++")
    # combining all matrices into one for speed
    MVP = camera_projection @ view_matrix @ model_matrix

    plt.figure(figsize=[xsize*5, ysize*5])
    plt.title("Same process using combined MVP matrix")
    ax = plt.gca()
    for mesh, color in zip(meshes, colors):
        print("=======")
        # Same effect can be achieved by calling
        pts_proj = [MVP @ homogenous(p) for p in mesh]
        print(f"{pts_proj=}")

        pts_cart = [project_by_matrix(MVP, p) for p in mesh]
        print(f"{pts_cart=}")

        for i in range(len(pts_cart)-1):
            plot_line(ax, pts_cart[i][0:2], pts_cart[i+1][0:2], color=color)

    plt.xlim([-xsize, xsize])
    plt.ylim([-ysize, ysize])

    plt.show()

if __name__ =="__main__":
    demo()