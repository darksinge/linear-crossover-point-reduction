"""
Crossover-Point Reduction is a strategy that takes as input a vector
of points in a 2D plane and outputs a vector of points that describes
the original vector in as few data points as possible.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math


class LinearCrossoverPointReduction(object):

    def __init__(self):
        pass

    def slope(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        b = x2 - x1
        if b == 0:
            return np.inf

        a = y2 - y1
        return a / b

    def y_intercept(self, p1, p2):
        """
        Find the y-intercept formed by points p1 and p2
        :param p1: first x,y-coordinate
        :param p2: second x,y-coordinate
        :return:
        """
        slope = self.slope(p1, p2)
        x1, y1 = p1
        y_int = y1 - slope * x1
        return 0.0, y_int

    def _point_of_intersection(self, a1, a2, b1, b2):
        """
        calculates the intersection point of two lines
        :param l1: first set of x,y-coordinates (ex: `l1 = [(1, 2), (3, 4)]`)
        :param l2: second set of x,y-coordinates
        :return:
        """
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return np.inf, np.inf
        return x / z, y / z

    def _target_in_bounds(self, target: float, x1, x2, x3, x4):
        if target == np.inf:
            return False
        return (x1 < target < x2) and (x3 < target < x4)

    def cross_reduce(self, v: np.ndarray, iterations=None):
        """
        "Smooths" the line out
        @param v: the vector that is being reduced
        @param iterations: the number of iterations to process `v`. The more iterations, the
                     more the original line will conform to it's "underlying" shape.
        """
        if not isinstance(v, np.ndarray):
            v = np.array(v)

        # ou = vector of over-under points
        # uo = vector of under-over points
        ou, uo = self.get_crossover_points(v)

        # u = np.empty(v.shape, dtype=float)
        u = v.copy()
        u[0] = ((ou[0][0] + uo[0][0]) / 2.0, (ou[0][1] + uo[0][1]) / 2.0)

        for i in range(1, len(v)):
            ou1 = ou[i-1]
            ou2 = ou[i]
            a1 = v[i-1]
            a2 = v[i]

            ou_intersects = self._point_of_intersection(ou1, ou2, a1, a2)
            if self._target_in_bounds(ou_intersects[0], ou1[0], ou2[0], a1[0], a2[0]):
                u[i] = ou_intersects
            else:
                # x = (ou[i-1][0] + uo[i-1][0]) / 2.0
                # y = (ou[i-1][1] + uo[i-1][1]) / 2.0
                # u[i] = (x, y)
                u[i] = None

        for i in range(1, len(v)-1):
            uo1 = uo[i-1]
            uo2 = uo[i]
            b1 = v[i]
            b2 = v[i+1]

            uo_intersects = self._point_of_intersection(uo1, uo2, b1, b2)
            if self._target_in_bounds(uo_intersects[0], uo1[0], uo2[0], b1[0], b2[0]):
                # assert u[i+1] is None
                u[i+1] = uo_intersects
            else:
                x = (ou[i][0] + uo[i][0]) / 2.0
                y = (ou[i][1] + uo[i][1]) / 2.0
                u[i+1] = (x, y)
                # u[i+1] = None

        assert len(u) == len(v)

        if iterations:
            return self.cross_reduce(u, iterations - 1)
        else:
            return u

    def get_crossover_points(self, v: np.ndarray):
        lenv = len(v)

        uo = np.empty(v.shape, dtype=float)
        for i in range(1, lenv):
            if self.slope(v[i], v[i - 1]) != 0:
                x = v[i][0]
                y = v[i-1][1]
                uo[i-1] = (x, y)
            else:
                uo[i-1] = (v[i][0], v[i][1])
        uo[-1] = (v[-1][0]+1, v[-1][1])

        ou = np.empty(v.shape, dtype=float)
        for i in range(0, lenv-1):
            if self.slope(v[i], v[i + 1]) != 0:
                x = v[i][0]
                y = v[i+1][1]
                ou[i] = (x, y)
            else:
                ou[i] = (v[i][0], v[i][1])
        ou[-1] = v[-1]  # "black magic"

        return ou, uo

    def point_reduce(self, v: np.ndarray):
        pass


def get_points(slope, intercept, start, end):
    x = []
    y = []

    start = int(start)
    end = int(end)

    for i in range(start, end):
        x.append(i)
        y.append(slope * i + intercept)

    return x, y


if __name__ == "__main__":
    # v = [0, 9, 6, 6, 5, 6, 6, 6, 7, 6, 6, 5]
    v = np.random.randint(0, 20, size=10)
    # v = [0] + v
    v = [p for p in zip(range(len(v)), v)]
    v = np.array(v, dtype=(float, 2))

    lcpr = LinearCrossoverPointReduction()

    vx = [x[0] for x in v]
    vy = [y[1] for y in v]
    slope, intercept, _, _, _ = stats.linregress(vx, vy)
    lx, ly = get_points(slope, intercept, vx[0], vx[-1])

    u = v.copy()
    i = 0
    done = False
    while not done:

        u = lcpr.cross_reduce(u, iterations=None)

        plt.plot(vx, vy, 'k.')

        ux = []
        uy = []

        for x, y in u:
            if math.isnan(x) or math.isnan(y):
                continue

            if x == 0 and y == 0:
                continue

            ux.append(x)
            uy.append(y)

        if len(ux) == 2:

            slope, intercept, _, _, _ = stats.linregress(ux, uy)
            lux, luy = get_points(slope, intercept, vx[0], vx[-1])
            # plt.plot(ux, uy, '')

            plt.plot(lx, ly, '-')  # line of best fit
            plt.plot(lux, luy, '--')  # something... else...

            plt.show()
            plt.clf()

        elif len(ux) < 2:
            break

        i += 1

