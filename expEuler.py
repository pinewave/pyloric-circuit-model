# 02/15/2016
# Author: Kun Tian
# exponential Euler method: Dayan and Abbott p.191-193

import numpy as np


def run(func, y0, tseries, params):

    nfeatures = len(y0)
    n = len(tseries)
    y = np.zeros(shape=(n, nfeatures))

    y[0] = y0
    for i, t in enumerate(tseries):
        if i <= n - 2:
            y[i + 1] = func(y[i], t, params)

            # if i % 100000 == 0:
            #     print(i)
    return y
