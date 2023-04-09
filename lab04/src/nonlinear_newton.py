import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from optimize import root


def var_f(x: np.array) -> np.array:
    return np.array([np.sin(0.5 * x[0] + x[1]) - 1.2 * x[0] - 1,
                     x[0]**2 + x[1]**2 - 1], dtype=np.double)


custom_solution_f, iter_cnt_f = root(var_f, np.array([-1, 0]))
custom_solution_s, iter_cnt_s = root(var_f, np.array([1, 2]))
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
