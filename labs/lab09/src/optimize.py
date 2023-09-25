import numpy as np
from typing import Callable


def deriv(func: Callable[[float], float],
          point: float,
          eps: float = 1e-5) -> float:
    return (func(point + eps) - func(point - eps)) / 2 * eps


def deriv2(func: Callable[[float], float],
           point: float,
           eps: float = 1e-5) -> float:
    return (func(point + eps) - 2 * func(point) + func(point - eps)) / eps ** 2


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


def jacobian(f: Callable[[np.array], np.array],
             x: np.array,
             f_cnt: int,
             eps: float = 1e-5) -> np.array:
    jac = np.zeros((f_cnt, len(x)), dtype=np.double)
    for i in range(len(x)):
        delta = np.zeros(len(x))
        delta[i] += eps
        jac[:, i] = (f(x + delta) - f(x - delta)) / (eps * 2)
    return jac


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


def newton_minimize_scal(func: Callable,
                         interval: tuple[float, float],
                         eps: float = 1e-5,
                         m: int = 1) -> tuple[float, int]:
    x, _ = interval
    cnt = 0
    while abs(func(x)) > eps:
        x -= m * deriv(func, x) / deriv2(func, x)
        cnt += 1
    return x, cnt


def fibonacci(func: Callable,
              interval: tuple[float, float],
              eps: float = 1e-5,
              minimize: bool = True) -> float:
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
    return (a + b) / 2
    
