from typing import Optional, Union
import numpy as np
import numpy.typing as npt

from sparse_matrix import SparseMatrix


def seidel(a: Union[npt.NDArray, SparseMatrix],
           b: npt.NDArray,
           x: Optional[npt.NDArray] = None,
           eps: float = 1e-9,
           max_iter: Optional[int] = None) -> tuple[npt.NDArray, int]:
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
        converge = np.linalg.norm(x_new - x, ord=np.inf) <= eps
        if max_iter:
            converge = iter_cnt >= max_iter
        x = x_new
    return x, iter_cnt


def simple_iterative(a: Union[npt.NDArray, SparseMatrix],
                     b: npt.NDArray,
                     x: Optional[npt.NDArray] = None,
                     eps: float = 1e-9,
                     max_iter: Optional[int] = None) -> tuple[npt.NDArray, int]:
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
