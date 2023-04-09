import numpy as np
from typing import Callable
import scipy.optimize
import matplotlib.pyplot as plt


def jacobian(f: Callable[[np.array], np.array],
             x: np.array,
             f_cnt: int,
             eps: float = 1e-3) -> np.array:
    jac = np.zeros((f_cnt, len(x)), dtype=np.double)
    for i in range(len(x)):
        delta = np.zeros(len(x))
        delta[i] += eps
        jac[:, i] = (f(x + delta) - f(x - delta)) / (eps * 2)
    return jac


def newton(f: Callable[[np.array], np.array],
           initial: np.array,
           eps: float = 1e-6) -> np.array:
    x = initial.astype(np.double)
    iter_cnt = 0
    f_cnt = len(f(initial))
    while np.linalg.norm(f(x)) > eps:
        x -= np.linalg.inv(jacobian(f, x, f_cnt)).dot(f(x))
        iter_cnt += 1
    return x, iter_cnt


def var_f(x: np.array) -> np.array:
    return np.array([np.sin(0.5 * x[0] + x[1]) - 1.2 * x[0] - 1,
                     x[0]**2 + x[1]**2 - 1], dtype=np.double)


custom_solution_f, iter_cnt_f = newton(var_f, np.array([-1, 0]))
custom_solution_s, iter_cnt_s = newton(var_f, np.array([1, 2]))
scipy_solution_f = scipy.optimize.fsolve(var_f, np.array([-1, 0]))
scipy_solution_s = scipy.optimize.fsolve(var_f, np.array([1, 2]))


print(f'Newton\'s method first solution: {custom_solution_f}')
print(f'Number of iterations: {iter_cnt_f}')
print(f'Newton\'s method second solution: {custom_solution_s}')
print(f'Number of iterations: {iter_cnt_s}')
print(f'Scipy solutions: {scipy_solution_f}, {scipy_solution_s}')


x, y = np.meshgrid(np.arange(-2, 2, 0.005), np.arange(-2, 2, 0.005))
plt.figure(figsize=(6, 5))
plt.contour(x, y, np.sin(0.5 * x + y) - 1.2 * x - 1, [0], colors=['blue'])
plt.contour(x, y, x**2 + y**2 - 1, [0], colors=['green'])
plt.savefig('plots/nonlinear_newton.png', dpi=300)
plt.show()
