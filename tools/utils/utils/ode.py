import numpy as np

import typing as tp


# constants
a = np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 1/2, 0, 0],
    [0, 0, 1, 0]
])
b = np.array([1/6, 1/3, 1/3, 1/6])
c = np.array([0, 1/2, 1/2, 1])


def extrapolate_adams(f: tp.Callable[[float, float], float],
                      y0: np.ndarray,
                      t0: np.ndarray,
                      t_end: float,
                      h: float) -> np.ndarray:
    n = int((t_end - t0[0]) / h)
    y = np.empty(n + 1)
    t = np.empty(n + 1)
    y[:2] = y0
    t[:2] = t0
    for i in range(1, n):
        y[i + 1] = y[i] + (h / 2) * (3 * f(t[i], y[i]) - f(t[i - 1], y[i - 1]))
    return y


def rk4_step(f: tp.Callable[[float, float], float],
             t: float,
             s: float,
             h: float) -> float:
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


def rk4_nsteps(f: tp.Callable[[float, float], float],
               y0: float,
               t0: float,
               t_end: float,
               h: float) -> np.ndarray:
    n = int((t_end - t0) / h)
    arr = np.empty((n + 1, 2))
    arr[:, 0] = np.linspace(t0, t_end, n + 1, endpoint=True)
    arr[0, 1] = y0

    for i in range(n):
        arr[i + 1, 1] = rk4_step(f,           # right part of SODE
                                 arr[i, 0],   # t_0
                                 arr[i, 1],   # s_0
                                 h)           # time step
    return arr[:, 1]


def runge_error(solver: tp.Callable[[tp.Any, tp.Any, tp.Any, tp.Any, tp.Any], np.ndarray],
                f: tp.Callable[[float, float], float],
                y0: float | np.ndarray,
                t0: float | np.ndarray,
                t_end: float,
                h: float,
                p: int) -> np.ndarray:
    normal_precision = solver(f, y0, t0, t_end, h)
    double_precision = solver(f, y0, t0, t_end, h / 2)
    errors = (double_precision[::2] - normal_precision) / (2 ** p - 1)
    return errors


def runge_error_step(solver_step: tp.Callable[[tp.Any, tp.Any, tp.Any, tp.Any], float | np.ndarray],
                     f: tp.Callable[[float, float], float],
                     y0: float | np.ndarray,
                     t0: float | np.ndarray,
                     h: float,
                     p: int) -> np.ndarray:
    y_h = solver_step(f, y0, t0, h)
    y_1_h2 = solver_step(f, y0, t0, h / 2)
    y_2_h2 = solver_step(f, y_1_h2, t0 + h / 2, h / 2)
    error = (y_h - y_2_h2) / (2 ** p - 1)
    return error


def eyler_step(f: tp.Callable[[float | np.ndarray, float | np.ndarray], float],
               y0: float | np.ndarray,
               t0: float | np.ndarray,
               h: float) -> float | np.ndarray:
    return y0 + h * f(t0, y0)


def eyler(f: tp.Callable[[float | np.ndarray, float | np.ndarray], float],
          y0: float | np.ndarray,
          t0: float | np.ndarray,
          t_end: float,
          h: float) -> np.ndarray:
    n = int((t_end - t0) / h)
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = (y[i] + h * f(t0 + i * h, y[i]))
    return y


def eyler_modified_step(f: tp.Callable[[float | np.ndarray, float | np.ndarray], float],
                        y0: float | np.ndarray,
                        t0: float | np.ndarray,
                        h: float) -> float | np.ndarray:
    y_pred = y0 + h * f(t0, y0)
    y_corr = y0 + h * (f(t0, y0) + f(t0 + h, y_pred)) / 2
    return y_corr


def eyler_modified(f: tp.Callable[[float | np.ndarray, float | np.ndarray], float],
                   y0: float | np.ndarray,
                   t0: float | np.ndarray,
                   t_end: float,
                   h: float) -> np.ndarray:
    n = int((t_end - t0) / h)
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = eyler_modified_step(f, y[i], t0 + i * h, h)
    return y


def eyler_adaptive(f: tp.Callable[[float | np.ndarray, float | np.ndarray], float],
                   y0: float | np.ndarray,
                   t0: float | np.ndarray,
                   t_end: float,
                   h0: float,
                   tol: float) -> tuple[list, list]:
    ys = [y0]
    ts = [t0]
    y_prev = y0
    t, h = t0, h0
    while t < t_end:
        y = eyler_modified_step(f, y_prev, t, h)
        err = abs(runge_error_step(eyler_modified_step, f, y_prev, t, h, 2))
        factor = min(max(tol / np.sqrt(2 * err), 0.3), 2)
        if factor >= 1:
            ys.append(y)
            y_prev = y
            t += h
            ts.append(t)
        h = 0.9 * h * factor
    return ys, ts
