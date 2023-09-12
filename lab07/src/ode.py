import numpy as np
from numba import njit


# constants
a = np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 1/2, 0, 0],
    [0, 0, 1, 0]
])
b = np.array([1/6, 1/3, 1/3, 1/6])
c = np.array([0, 1/2, 1/2, 1])


@njit
def eyler(f, y0, t0, h, N):
    y = np.empty(N + 1)
    y[0] = y0
    for i in range(N):
        y[i + 1] = (y[i] + h * f(t0 + i * h, y[i]))
    return y


def rk_step(f, t, s, h):
    ss = 4
    k = [f(t, s)]
    for i in range(1, ss):
        diff = s
        for j in range(i):
            diff += h * a[i, j] * k[j]
        k.append(f(t + c[i] * h, diff))
    diff = s
    for i in range(ss):
        diff += h * b[i] * k[i]
    return diff


def rk_nsteps(f, t0, y0, t_end, h):
    n = int((t_end - t0) / h)
    arr = np.empty((n + 1, 2))
    arr[:, 0] = np.linspace(t0, t_end, n + 1)
    arr[0, 1] = y0

    for i in range(n):
        arr[i + 1, 1] = rk_step(f,           # right part of SODE
                                arr[i, 0],   # t_0
                                arr[i, 1],   # s_0
                                h)           # time step
    return arr



