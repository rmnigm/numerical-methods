import numpy as np
from typing import Callable


def deriv(func: Callable[[float], float],
          point: float,
          eps: float = 1e-5) -> float:
    return (func(point + eps) - func(point - eps)) / (2 * eps)


def deriv2(func: Callable[[float], float],
           point: float,
           eps: float = 1e-5) -> float:
    return (func(point + eps) - 2 * func(point) + func(point - eps)) / (eps ** 2)


def grad(f: Callable[[np.array], np.array],
         x: np.array,
         eps: float = 1e-5) -> np.array:
    dim = len(x)
    grad_vector = np.zeros((dim, ), dtype=np.double)
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] += eps
        grad_vector[i] = (f(x + delta) - f(x - delta)) / (eps * 2)
    return grad_vector


def hessian(f: Callable[[np.array], np.array],
            x: np.array,
            eps: float = 1e-5) -> np.array:
    dim = len(x)
    hess = np.zeros((dim, dim), dtype=np.double)
    for i in range(dim):
        i_d = np.zeros(dim)
        i_d[i] += eps
        for j in range(dim):
            j_d = np.zeros(dim)
            j_d[j] += eps
            hess[i, j] = (f(x - i_d - j_d) - f(x + i_d - j_d)
                          - f(x - i_d + j_d) + f(x + i_d + j_d)
                          ) / (4 * eps ** 2)
    return hess


def point_in_area(point, bbox):
    return bbox[0][0] < point[0] < bbox[1][0] and bbox[0][1] < point[1] < bbox[1][1]


def newton_optimize_vec(func: Callable[[np.array], np.array],
                        bbox: tuple[np.array, np.array],
                        start: np.array,
                        eps: float = 1e-5,
                        minimize: bool = True) -> tuple[np.array, int]:
    x = start.astype(np.double)
    point_grad = grad(func, x)
    iter_cnt = 0
    while point_in_area(x, bbox) and np.linalg.norm(point_grad) > eps:
        step = np.linalg.inv(hessian(func, x)).dot(point_grad)
        x -= step if minimize else -step
        point_grad = grad(func, x)
        iter_cnt += 1
    x[0] = max(min(bbox[1][0], x[0]), bbox[0][0])
    x[1] = max(min(bbox[1][1], x[1]), bbox[0][1])
    return x, iter_cnt


def conjucate_grad_minimize(func: Callable[[np.array], np.array],
                            start: np.array,
                            eps: float = 1e-5) -> tuple[np.array, int]:
    x = start.astype(np.double)
    x_prev, h_prev = None, None
    iter_cnt = 0
    h = -grad(func, x)
    while np.linalg.norm(grad(func, x)) > eps:
        h = -grad(func, x)
        if h_prev is not None and x_prev is not None:
            beta = (np.linalg.norm(grad(func, x)) / np.linalg.norm(grad(func, x_prev))) ** 2
            h += beta * h_prev
        alpha, _ = newton_optimize_scal(lambda a: func(x + a * h),
                                        interval=(-5, 5),
                                        start=0,
                                        eps=eps * 1e-2)
        x_prev = x.copy()
        x += alpha * h
        h_prev = h.copy()
        iter_cnt += 1

    return x, iter_cnt


def conjucate_grad_quadratic_minimize(func: Callable[[np.array], np.array],
                                      func_matrix: np.array,
                                      start: np.array,
                                      eps: float = 1e-5) -> tuple[np.array, int]:
    x = start.astype(np.double)
    h_prev = None
    iter_cnt = 0
    h = -grad(func, x)
    while np.linalg.norm(grad(func, x)) > eps:
        h = -grad(func, x)
        if h_prev is not None:
            beta = ((func_matrix @ h_prev) @ grad(func, x)) / ((func_matrix @ h_prev) @ h_prev)
            h += beta * h_prev
        alpha, _ = newton_optimize_scal(lambda a: func(x + a * h),
                                        interval=(-10, 10),
                                        start=0,
                                        eps=eps * 1e-2)
        x += alpha * h
        h_prev = h.copy()
        iter_cnt += 1

    return x, iter_cnt


def newton_optimize_scal(func: Callable,
                         interval: tuple[float, float],
                         start: float,
                         eps: float = 1e-5,
                         minimize: bool = True) -> tuple[float, int]:
    (a, b), x = interval, start
    cnt = 0
    while a <= x <= b and abs(deriv(func, x)) > eps:
        step = deriv(func, x) / deriv2(func, x)
        x += -step if minimize else step
        cnt += 1
    x = min(b, max(a, x))
    return x, cnt


def fibonacci(func: Callable,
              interval: tuple[float, float],
              eps: float = 1e-5,
              minimize: bool = True) -> tuple[float, int]:
    a, b = interval
    numbers, f = [1, 1], 2
    d = b - a
    while f <= (b - a) / eps:
        numbers.append(f)
        f = numbers[-1] + numbers[-2]
    n = len(numbers)
    for k in range(1, n):
        d *= numbers[n - k - 1] / numbers[n - k]
        x1 = b - d
        x2 = a + d
        cond = func(x1) <= func(x2) if minimize else func(x1) >= func(x2)
        if cond:
            b = x2
        else:
            a = x1
    return (a + b) / 2, n
    
