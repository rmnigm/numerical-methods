import numpy as np
import typing as tp

from .diff import deriv, deriv2, grad, hessian, jacobian


def point_in_area(point: np.ndarray, bbox: tp.Tuple[np.ndarray, np.ndarray]):
    return bbox[0][0] < point[0] < bbox[1][0] and bbox[0][1] < point[1] < bbox[1][1]


def fibonacci(func: tp.Callable,
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


def newton_root_vec(f: tp.Callable[[np.ndarray], np.ndarray],
                    initial: np.ndarray,
                    eps: float = 1e-6) -> tp.Tuple[np.ndarray, int]:
    x = initial.astype(np.double)
    iter_cnt = 0
    f_cnt = len(f(initial))
    while np.linalg.norm(f(x)) > eps:
        x -= np.linalg.inv(jacobian(f, x, f_cnt)).dot(f(x))
        iter_cnt += 1
    return x, iter_cnt


def newton_optimize_vec(func: tp.Callable[[np.ndarray], np.ndarray],
                        bbox: tp.Tuple[np.ndarray, np.ndarray],
                        start: np.ndarray,
                        eps: float = 1e-5,
                        minimize: bool = True) -> tuple[np.ndarray, int]:
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


def newton_root_scal(func: tp.Callable,
                     interval: tuple[float, float],
                     eps: float = 1e-5,
                     m: int = 1) -> tuple[float, int]:
    x, _ = interval
    cnt = 0
    while abs(func(x)) > eps:
        x -= m * func(x) / deriv(func, x)
        cnt += 1
    return x, cnt


def newton_optimize_scal(func: tp.Callable[[tp.Any], tp.Any],
                         interval: tuple[float, float],
                         start: float,
                         eps: float = 1e-5,
                         minimize: bool = True) -> tuple[tp.Any, int]:
    (a, b), x = interval, start
    cnt = 0
    while a <= x <= b and abs(deriv(func, x)) > eps:
        step = deriv(func, x) / deriv2(func, x)
        x += -step if minimize else step
        cnt += 1
    x = min(b, max(a, x))
    return x, cnt


def bisection_root_scal(f: tp.Callable,
                        interval: tuple[float, float],
                        eps: float = 1e-5) -> float:
    a, b = interval
    while abs(a - b) > 2 * eps:
        x = (a + b) / 2
        a_val, x_val = f(a), f(x)
        if a_val * x_val <= 0:
            b = x
        else:
            a = x
    return (a + b) / 2


def conjucate_grad_minimize(func: tp.Callable[[np.ndarray], np.ndarray],
                            start: np.ndarray,
                            eps: float = 1e-5) -> tuple[np.ndarray, int]:
    x = start.astype(np.double)
    x_prev, h_prev = None, None
    iter_cnt = 0
    h = -grad(func, x)
    while np.linalg.norm(grad(func, x)) > eps:
        h = -grad(func, x)
        if h_prev is not None and x_prev is not None:
            denominator = np.linalg.norm(grad(func, x_prev))
            beta = (np.linalg.norm(grad(func, x)) / denominator) ** 2
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


def conjucate_grad_quadratic_minimize(func: tp.Callable[[np.ndarray], np.ndarray],
                                      func_matrix: np.ndarray,
                                      start: np.ndarray,
                                      eps: float = 1e-5) -> tuple[np.ndarray, int]:
    x = start.astype(np.double)
    h_prev = None
    iter_cnt = 0
    h = -grad(func, x)
    while np.linalg.norm(grad(func, x)) > eps:
        h = -grad(func, x)
        if h_prev is not None:
            denominator = ((func_matrix @ h_prev) @ h_prev)
            beta = ((func_matrix @ h_prev) @ grad(func, x)) / denominator
            h += beta * h_prev
        alpha, _ = newton_optimize_scal(lambda a: func(x + a * h),
                                        interval=(-10, 10),
                                        start=0,
                                        eps=eps * 1e-2)
        x += alpha * h
        h_prev = h.copy()
        iter_cnt += 1
    
    return x, iter_cnt
