from typing import Optional, Union

import numpy as np

from .sparse_matrix import SparseMatrix


def seidel(a: Union[np.ndarray, SparseMatrix],
           b: np.ndarray,
           x: Optional[np.ndarray] = None,
           eps: float = 1e-9,
           max_iter: Optional[int] = None) -> tuple[np.ndarray, int]:
    """
    Seidel solver for system of linear equations.

    :param a: np array or sparse matrix, matrix form of the system
    :param b: np array, right part vector of the system
    :param x: np array, initial point for solution
    :param eps: float, precision by norm
    :param max_iter: int, number of max iterations
    """
    n = len(a)
    x = x if x is not None else np.zeros(n)
    converge = False
    iter_cnt = 0
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(a[i, j] * x_new[j] for j in range(i))
            s2 = sum(a[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / a[i, i]
        iter_cnt += 1
        converge = bool(np.linalg.norm(x_new - x, ord=np.inf) <= eps)
        if max_iter:
            converge = iter_cnt >= max_iter
        x = x_new
    return x, iter_cnt


def simple_iterative(a: np.ndarray,
                     b: np.ndarray,
                     x: Optional[np.ndarray] = None,
                     eps: float = 1e-9,
                     max_iter: int = 1000) -> tuple[np.ndarray, int]:
    x = x if x is not None else np.zeros(len(a))
    L = np.tril(a, -1)
    U = np.triu(a, 1)
    D = np.diag(np.diag(a))
    B = np.linalg.inv(L + D).dot(-U)
    c = np.linalg.inv(L + D).dot(b)
    x_new = x.copy()
    for i in range(max_iter):
        x_new = x.copy()
        x_new = B.dot(x_new) + c
        if np.linalg.norm(x - x_new, ord=np.inf) <= eps:
            return x_new, i
        x = x_new
    return x_new, max_iter


def lstsq(X, Y, m):
    b = np.zeros(m)
    G = np.zeros((m, m))
    for j in range(m):
        b[j] = sum(y * x ** j for y, x in zip(Y, X))
        for k in range(m):
            G[j, k] = sum(x ** (k + j) for x in X)
    return np.linalg.solve(G, b)


def tridiagonal_solve(a: np.ndarray,
                      b: np.ndarray,
                      c: np.ndarray,
                      d: np.ndarray) -> np.ndarray:
    for it in range(1, len(d)):
        mc = a[it - 1] / b[it - 1]
        b[it] = b[it] - mc * c[it - 1]
        d[it] = d[it] - mc * d[it - 1]
    xc = b
    xc[-1] = d[-1] / b[-1]
    for il in range(len(d) - 2, -1, -1):
        xc[il] = (d[il] - c[il] * xc[il + 1]) / b[il]
    return xc
