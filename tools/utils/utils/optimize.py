import numpy as np
from typing import Callable

from .diff import deriv, grad, hessian, jacobian


def newton_root_vec(f: Callable[[np.array], np.array],
                    initial: np.array,
                    eps: float = 1e-6) -> np.array:
    x = initial.astype(np.double)
    iter_cnt = 0
    f_cnt = len(f(initial))
    while np.linalg.norm(f(x)) > eps:
        x -= np.linalg.inv(jacobian(f, x, f_cnt)).dot(f(x))
        iter_cnt += 1
    return x, iter_cnt


def newton_minimize_vec(f: Callable[[np.array], np.array],
                        initial: np.array,
                        eps: float = 1e-6) -> np.array:
    x = initial.astype(np.double)
    point_grad = grad(f, x)
    iter_cnt = 0
    while np.linalg.norm(point_grad) > eps:
        x -= np.linalg.inv(hessian(f, x)).dot(point_grad)
        point_grad = grad(f, x)
        iter_cnt += 1
    return x, iter_cnt


def newton_root_scal(func: Callable,
                     interval: tuple[float, float],
                     eps: float = 1e-5,
                     m: int = 1) -> tuple[float, int]:
    x, _ = interval
    cnt = 0
    while abs(func(x)) > eps:
        x -= m * func(x) / deriv(func, x)
        cnt += 1
    return x, cnt


def bisection_root_scal(f: Callable,
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
