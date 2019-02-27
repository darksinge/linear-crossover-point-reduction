"""
Crossover-Point Reduction is a strategy that takes as input a vector
of points in a 2D plane and outputs a vector of points that describes
the original vector in as few data points as possible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# from operator import xor
import pandas as pd
from functools import reduce
# from scipy.spatial.distance import cosine

from dataloader import group_events, save_plot


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

    def point_average(self, x1, y1, x2, y2):
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

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

    def cosine(self, u: (float, float), v: (float, float)) -> float:
        """
        Returns the cosine angle between two 2D vectors in degrees
        :param u: the first vector
        :param v: the second vector
        :return: the angle between `u` and `v` in degrees
        """
        x1, y1 = u
        x2, y2 = v
        udotv = x1 * x2 + y1 * y2
        umag = np.sqrt(x1 ** 2 + y1 ** 2)
        vmag = np.sqrt(x2 ** 2 + y2 ** 2)
        cos = udotv / (umag * vmag)
        arccos = np.arccos(cos)
        return np.rad2deg(arccos)

    def std_dev(self, v: np.ndarray):
        """
        Returns the standard deviation of an array
        """
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=(float, 2))

        u = np.array([x[1] for x in v])
        return u.std()

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
        "Smooths" the line out. The "reduction" in this step has to do with
        reducing the local maximum and local minimum data points.

        :param v: the vector that is being reduced
        :param iterations: the number of iterations to process `v`. The more iterations, the
                     more the original line will conform to it's "underlying" shape.
        :return the cross-reduced numpy array and the difference of `v's and `u's standard deviations
        """
        if not isinstance(v, np.ndarray):
            v = np.array(v)

        # ou = vector of over-under points
        # uo = vector of under-over points
        ou, uo = self.get_crossover_points(v)

        u = v.copy()

        for i in range(1, len(v)-1):

            # get the points of intersection for `ou->v' and `uo->v'
            ou_ipoint = self._point_of_intersection(ou[i-1], ou[i], v[i-1], v[i])
            uo_ipoint = self._point_of_intersection(uo[i-1], uo[i], v[i], v[i+1])

            # check if `ou_point' and `v[i]' intersect between the x-value of
            # the previous point and the x-value of the current point.
            ou_intersects = self._target_in_bounds(ou_ipoint[0], ou[i-1][0], ou[i][0], v[i-1][0], v[i][0])
            uo_intersects = self._target_in_bounds(uo_ipoint[0], uo[i-1][0], uo[i][0], v[i][0], v[i+1][0])

            # assert not(ou_intersects and uo_intersects)

            if ou_intersects and uo_intersects:
                u[i] = self.point_average(v[i][0], uo[i][1], v[i][0], ou[i][1])
            elif ou_intersects:
                u[i] = ou_ipoint

            elif uo_intersects:
                u[i] = uo_ipoint

            else:
                # u[i] = self.point_average(uo[i][0], uo[i][1], ou[i][0], ou[i][1])
                u[i] = v[i]

        # u = self.point_reduce(u, v)

        if iterations:
            return self.cross_reduce(u, iterations - 1)
        else:
            deviation = abs(self.std_dev(v) - self.std_dev(u))
            return u, deviation

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

    def point_reduce(self, u: np.ndarray, v: np.ndarray):
        if u.shape != v.shape:
            raise ValueError("mismatched dimensions")

        q = []
        # find the union of `u' and `v', store in `q`
        for p1, p2 in zip(u, v):
            # if p1[0] == p2[0] and p1[1] == p2[1]:
            if tuple(p1) == tuple(p2):
                q.append(tuple(p1))

        # remove the middle of 3 consecutive points in 'q'
        tolerance = 0.001
        n = len(q) - 1
        i = 1
        while i < n:

            a: float = q[i-1][1]  # first point's y-value
            b: float = q[i][1]    # middle point's y-value
            c: float = q[i+1][1]  # last point's y-value

            # check the distance between the first and middle, and the middle and last
            if abs((b - a) + (c - b)) < 2*tolerance:
                q.pop(i)
                n -= 1
            else:
                i += 1

        # take the average of 2 consecutive points
        n = len(q)
        i = 1
        while i < n:
            x1, y1 = q[i - 1]
            x2, y2 = q[i]

            if y1 == y2:
                x, y = (x1+x2) / 2., (y1+y2) / 2.
                q[i-1] = (x, y)
                q.pop(i)
                n -= 1
            else:
                i += 1

        return np.array(q, dtype=(float, 2))

    def point_reduce_vector(self, u, max_y_variation=0.5):
        pleft = []
        pright = []
        for i in range(1, len(u)):
            variation = abs(u[i-1][1] - u[i][1])
            if variation > max_y_variation:
                pleft.append(u[i-1])
                pright.append(u[i])

        p = pleft + pright
        p.sort(key=lambda x: x[0])

        return p


def main():

    # x = np.linspace(-np.pi, np.pi, 10)
    # y = [0, 9, 6, 6, 5, 6, 6, 6, 7, 6, 6, 5, 0]
    # y = [0, 20, 20, 1, 2, 0]
    # y = np.sin(x)

    # groups = group_events()
    # groups = list(reduce(lambda a, b: a + b, groups))
    # for i, df in enumerate(groups):  # type: pd.DataFrame
    #     df.to_csv("./data/output/csv/%d.csv" % i)
    #     save_plot(df, "./data/images/plots", fname="%d.png" % i)

    df = pd.read_csv("./data/output/csv/79.csv")
    # df = groups[18]

    y = [x for x in df['pulse']][:100]
    x = [i for i in range(len(y))]

    # v = np.array([p for p in zip(range(len(y_vals)), y_vals)], dtype=(float, 2))
    v = np.array([p for p in zip(x, y)], dtype=(float, 2))

    lcpr = LinearCrossoverPointReduction()

    u = v.copy()
    done = False
    n = 0
    figure: Figure = plt.figure()
    figure.dpi = 300
    while not done:
        u, std_err = lcpr.cross_reduce(u, iterations=None)

        # plt.plot([x[0] for x in v], [x[1] for x in v], '-', linewidth=2)
        # plt.plot([x[0] for x in u], [x[1] for x in u], 'o--', linewidth=2)
        # plt.show()
        # plt.clf()

        # for j in range(2, len(u)):
        #     xdiff, ydiff = u[j-2]
        #     U = (u[j][0] - xdiff, u[j][1] - ydiff)
        #     V = (u[j-1][0] - xdiff, u[j-1][1] - ydiff)
        #     angle = lcpr.cosine(U, V)
        #     if angle > 10.0:
        #         U = (U[0] + xdiff, U[1] + ydiff)
        #         V = (V[0] + xdiff, V[1] + ydiff)
        #         print(U, V)
        #     print(angle)

        n += 1
        if len(u) <= 4:
            done = True

        if std_err < 0.01:
            print('converged in %d cycle%s' % (n, "s" if n > 1 else ""))
            done = True

    u = lcpr.point_reduce_vector(u, 0.5)
    plt.plot([x[0] for x in v], [x[1] for x in v], '-', linewidth=2)
    plt.plot([x[0] for x in u], [x[1] for x in u], 'o--', linewidth=2)
    plt.show()
    plt.clf()

    # plt.savefig("./data/images/plots/~.png", dpi=1000)


if __name__ == '__main__':
    main()
