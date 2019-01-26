"""
Crossover-Point Reduction is a strategy that takes as input a vector
of points in a 2D plane and outputs a vector of points that describes
the original vector in as few data points as possible.
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import xor


class LinearCrossoverPointReduction(object):

    def __init__(self):
        pass

    def _slope(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        b = x2 - x1
        if b == 0:
            return np.inf

        a = y2 - y1
        return a / b

    def _y_intercept(self, p1, p2):
        """
        Find the y-intercept formed by points p1 and p2
        :param p1: first x,y-coordinate
        :param p2: second x,y-coordinate
        :return:
        """
        slope = self._slope(p1, p2)
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

        u = v.copy()

        for i in range(1, len(v)):

            ou_ipoint = self._point_of_intersection(ou[i-1], ou[i], v[i-1], v[i])
            uo_ipoint = self._point_of_intersection(uo[i-1], uo[i], v[i-1], v[i])

            ou_intersects = self._target_in_bounds(ou_ipoint[0], ou[i-1][0], ou[i][0], v[i-1][0], v[i][0])
            uo_intersects = self._target_in_bounds(uo_ipoint[0], uo[i-1][0], uo[i][0], v[i-1][0], v[i][0])

            assert not(ou_intersects and uo_intersects)

            if ou_intersects:
                u[i] = ou_ipoint
            elif uo_intersects:
                u[i] = uo_ipoint
            else:
                u[i] = v[i]

        if iterations:
            return self.cross_reduce(u, iterations - 1)
        else:
            return u

    def get_crossover_points(self, v: np.ndarray):
        lenv = len(v)

        uo = np.empty(v.shape, dtype=float)
        ou = np.empty(v.shape, dtype=float)

        for i in range(1, lenv):

            if self._slope(v[i-1], v[i]):
                uo[i-1] = (v[i][0], v[i-1][1])
                ou[i-1] = (v[i-1][0], v[i][1])

            else:
                uo[i-1] = v[i]
                ou[i-1] = v[i-1]

        uo[lenv-1] = (v[-1][0]+1, v[-1][1])
        ou[lenv-1] = v[-1]

        return ou, uo

    def point_reduce(self, v: np.ndarray):
        pass


if __name__ == "__main__":
    y_vals = [0, 9, 6, 6, 5, 6, 6, 6, 7, 6, 6, 5, 0]
    v = np.array([p for p in zip(range(len(y_vals)), y_vals)], dtype=(float, 2))

    lcpr = LinearCrossoverPointReduction()

    u = lcpr.cross_reduce(v, iterations=100)

    plt.plot([x[0] for x in v], [x[1] for x in v], 'b')
    plt.plot([x[0] for x in u], [x[1] for x in u], 'r')
    plt.show()

