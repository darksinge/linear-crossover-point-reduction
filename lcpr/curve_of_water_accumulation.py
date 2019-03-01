import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


def plot_cwa(u: np.array, v: np.array):

    u_y_values = [x[1] for x in u]
    v_y_values = [x[1] for x in v]

    u_vol = []
    v_vol = []
    y_diffs = []
    u_vol_total = 0
    v_vol_total = 0

    for u_y, v_y in zip(u_y_values, v_y_values):
        u_vol_total += u_y
        v_vol_total += v_y
        u_vol.append(u_vol_total)
        v_vol.append(v_vol_total)

        y_diffs.append(abs(u_vol_total - v_vol_total))

    x = [x[0]*4 for x in u]

    host = host_subplot(111, axes_class=AA.Axes)

    par = host.twinx()

    host.set_xlim(0, max(x))
    host.set_ylim(0, max(max(u_vol), max(v_vol))+10)
    host.set_xlabel("Time (s)")
    host.set_ylabel("Accumulated Volume")

    par.set_xlim(0, max(x))
    par.set_ylim(0, max(y_diffs)+(max(y_diffs)*0.1))
    par.set_ylabel("Volume Difference")

    p1, = host.plot(x, u_vol)
    p2, = host.plot(x, v_vol)
    p3, = par.plot(x, y_diffs)

    # host.axis["left"].label.set_color(p1.get_color())


    # plt.plot(x, u_vol)
    # plt.plot(x, v_vol)
    # plt.plot(x, y_diffs)
    plt.draw()
    plt.show()



