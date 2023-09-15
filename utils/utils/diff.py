import numpy as np
from typing import Callable


def deriv(func: Callable[[float], float],
          point: float,
          eps: float = 1e-5) -> float:
    return (func(point + eps) - func(point - eps)) / 2 * eps


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
