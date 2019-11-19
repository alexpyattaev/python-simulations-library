
from debug_log import *
import numpy as np
from lib import hexgrid


__author__ = 'Alex Pyattaev'


class Spatial_Hash_Grid(object):
    def __init__(self, cell_size=10, grid_size=50, object_classes=("TX", "RX"), layout=None, **kwargs):
        assert (grid_size % 2 == 0), "The grid size must be even integer number"
        self.object_grids = dict()
        #Create object grids for all objects of interest
        for c in object_classes:
            self.object_grids[c] = np.array([[set() for x in range(grid_size)] for y in range(grid_size)], dtype=object)
        self.wrap_pointers = np.array([[[-1,-1] for x in range(grid_size)] for y in range(grid_size)], dtype=int)
        self.wrap_transforms = np.array([[0 for x in range(grid_size)] for y in range(grid_size)], dtype=int)

        self.g_half = grid_size // 2
        self.cell_size = cell_size
        self.items = {}

        self.layout = layout
        self.canvas = None
        self.axes = None
        if layout == "7CELL":
            self.R = kwargs["cell_R"]
            cells = hexgrid.hexgrid_cells(7)
            for c in cells:
                poly = hexgrid.hexgrid_polygon(c[0], c[1], r)
        elif layout == "BOX":
            pass
        elif layout is None:
            pass
        else:
            raise ValueError("Unsupported layout " + str(layout))

    def change_cell(self, pos:np.ndarray, item:object, object_class:str):

        # Is this really needed? Maybe assume that cells never change during lifetime of a session?
        try:
            old_cell = self.items[(object_class, item)]
            old_cell.remove(item)
        except KeyError:
            pass

        c = self[object_class, pos[0:2]]
        print("Adding to cell {}".format(c))
        c.add(item)
        self.items[(object_class, item)] = c

    def __getitem__(self, item ) -> set:
        """
        Return the cell pointer based on coordinates in meters
        :param pos_m: position tuple or array in meters
        """
        object_class, pos_m = item
        x_m, y_m = pos_m
        x = int(x_m / self.cell_size) + self.g_half
        y = int(y_m / self.cell_size) + self.g_half
        return self.object_grids[object_class][x, y]

    def init_wraparound(self):
        raise NotImplementedError()

    def select_circle(self, x_m: float, y_m: float, R_m: float, object_class:str, coord_mode=False)->set:
        """
        Returns an iterator over cells in the hash that are around a given center point.

        if coord_mode=True the return changes to raw coordinate pairs

        :param x_m: X coordinate of the source in meters
        :param y_m: Y coordinate of the source in meters
        :param R_m: Radius of coverage in meters
        :param coord_mode: set True to return raw cell coords.
        :returns: iterator of cell pointers to be visited
        """

        x_m = int(x_m / self.cell_size) + self.g_half
        y_m = int(y_m / self.cell_size) + self.g_half
        R = int(np.ceil(R_m / self.cell_size))
        gmax = self.g_half * 2
        if R < 2:
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if 0 <= x_m + dx < gmax and 0 <= y_m + dy < gmax:
                        if coord_mode:
                            yield [x_m + dx, y_m + dy]
                        else:
                            yield self.object_grids[object_class][x_m + dx, y_m + dy]
                    else:
                        pass
        else:
            # Bresenham's algo borrowed from http://members.chello.at/easyfilter/bresenham.html
            x = -R
            y = 0
            err = 2 - 2 * R

            draw = True
            while True:
                if draw:
                    for dx in np.arange(x, -x + 1):
                        if gmax > x_m - dx >= 0 and y_m + y < gmax:
                            # I and II Quadrant
                            if coord_mode:
                                yield [x_m - dx, y_m + y]
                            else:
                                yield self.object_grids[object_class][x_m - dx, y_m + y]

                        if 0 < y <= y_m and gmax > x_m + dx >= 0:
                            # III and IV Quadrant
                            if coord_mode:
                                yield [x_m + dx, y_m - y]
                            else:
                                yield self.object_grids[object_class][x_m + dx, y_m - y]
                draw = False
                r = err
                if (r <= y):
                    y += 1
                    err += y * 2 + 1  # e_xy+e_y < 0
                    draw = True
                if r > x or err > y:
                    x += 1
                    err += x * 2 + 1  # e_xy+e_x > 0 or no 2nd y-step
                if x > 0:
                    break

    def select_line(self, x_0: float, y_0: float, x_1: float,y_1: float, object_class:str, coord_mode=False) -> set:
        """
        Returns an iterator over cells in the hash that are around a given center point.

        if coord_mode=True the return changes to raw coordinate pairs

        :param x_0: X coordinate of the source in meters
        :param y_0: Y coordinate of the source in meters
        :param x_1: X coordinate of the dest in meters
        :param y_1: Y coordinate of the dest in meters
        :param coord_mode: set True to return raw cell coords.
        :returns: iterator of cell pointers to be visited
        """


        # Bresenham's algo borrowed from http://members.chello.at/easyfilter/bresenham.html
        dx = abs(x_1 - x_0)
        sx = [-1, 1][x_0 < x_1]
        dy = -abs(y_1 - y_0)
        sy = [-1, 1][y_0 < y_1]
        err = dx + dy
        while True:
            if coord_mode:
                yield (x_0, y_0)
            else:
                yield self.object_grids[object_class][x_0, y_0]

            if x_0 == x_1 and y_0 == y_1:
                break
            e2 = 2 * err
            if (e2 >= dy):
                err += dy
                x_0 += sx

            if (e2 <= dx):
                err += dx
                y_0 += sy

    def draw_cells(self, cell_list):

        patches = []
        for x, y in cell_list:
            t = (np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) +
                 np.array([x - self.g_half, y - self.g_half])) * self.cell_size
            patches.append(Polygon(t, closed=True))
        colors = np.full(len(patches), 50, dtype=int)
        patches = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
        patches.set_array(colors)
        self.axes.add_collection(patches)

    def plot_self(self):
        if self.canvas is None:
            self.canvas = plt.figure()
        self.canvas.clf()

        self.axes = self.canvas.add_subplot(1, 1, 1)

        plt.axis('equal')
        patches = []
        colors = []
        g_half = self.g_half
        self.axes.set_xlim(np.array([-g_half * self.cell_size, g_half * self.cell_size]) * 1.2)
        self.axes.set_ylim(np.array([-g_half * self.cell_size, g_half * self.cell_size]) * 1.2)
        for x in range(-g_half, g_half):
            for y in range(-g_half, g_half):
                t = (np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) + np.array([x, y])) * self.cell_size
                patches.append(Polygon(t, closed=True))
                cnt = 0
                for c, g in self.object_grids.items():
                    cnt += len(g[x + g_half, y + g_half])
                if cnt > 0:
                    colors.append(50)
                else:
                    colors.append(0)
        patches = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
        colors = np.array(colors)
        patches.set_array(np.array(colors))
        self.axes.add_collection(patches)

        if self.layout is not None:
            # cells = hexgrid.hexgrid_cells(19)
            cells = hexgrid.hexgrid_cells(7)
            for c in cells:
                poly = hexgrid.hexgrid_polygon(c[0], c[1], r)
                self.canvas.plot(poly[:, 0], poly[:, 1])
            phi = np.arange(0, 2 * np.pi, 0.01)
            self.canvas.plot(np.cos(phi) * 2 * r, np.sin(phi) * 2 * r)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    grid = Spatial_Hash_Grid(10, 50)
    class Foo():
        pass

    grid.change_cell(pos=np.array([20,30,0]), item="TEST", object_class="TX")
    grid.change_cell(pos=np.array([20, 50, 0]), item="TEST", object_class="TX")
    grid.change_cell(pos=np.array([20, 70, 0]), item="TEST", object_class="RX")
    cells = list(grid.select_circle(30,20,30, object_class="TX", coord_mode=True))

    grid.plot_self()
    grid.draw_cells(cells)

    for c in grid.select_circle(30, 20, 30, object_class="TX"):
        print(c)
    r = 100
    plt.figure()


    # cells = hexgrid.hexgrid_cells(19)
    cells = hexgrid.hexgrid_cells(7)
    for c in cells:
        poly = hexgrid.hexgrid_polygon(c[0], c[1], r)
        plt.plot(poly[:, 0], poly[:, 1])

    phi = np.arange(0, 2 * np.pi, 0.01)
    plt.plot(np.cos(phi) * 2 * r, np.sin(phi) * 2 * r)
    plt.show()
