__author__ = 'Alex Pyattaev'
import numpy as np
from scipy.interpolate import griddata

def surfplot_heatmap(pos, data, grid_size:float, N_pts:int=100, fill:float=0):
    """
    Prepares interpolated data which is only known at select points given by pos,
    over a grid given by grid_size, sampled uniformly N_pts x N_pts times

    :param pos: positions of points with known data of shape (n,2)
    :param data: data values (n)
    :param grid_size: grid size (limits)
    :param fill: what to use if data value is undefined
    :param N_pts: number of interpolated points
    :return data suitable for mesh plotting (X, Y, Z)

    Example usage:
    X,Y,Z = surfplot_heatmap(positions, coverage_map, grid_size=50.0, fill=-180)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    cont = ax.plot_contour(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    """
    #xlin = np.linspace(-grid_size/2, grid_size/2, N_pts)
    #ylin = np.linspace(-grid_size/2, grid_size/2, N_pts)
    step = grid_size/N_pts
    X,Y = np.mgrid[-grid_size/2:grid_size/2:step, -grid_size/2:grid_size/2:step]
    #http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.griddata.html
    Z = griddata(pos, data, (X, Y), method='linear',fill_value=fill)
    return X,Y,Z
