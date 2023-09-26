import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from ode import eyler, rk4_nsteps, runge_error


@njit
def f(t, y):
    return - y / t + 3 * t


@njit
def analytical_solution(t):
    return t ** 2


t0, t_end, y0 = 1, 2, 1
h = 0.1
N = int((t_end - t0) / h)

t_values = np.array([t0 + i * h for i in range(N + 1)])

analytical_values = np.array([analytical_solution(t) for t in t_values])
eyler_values = np.array(eyler(f, y0, t0, t_end, h))

rk_values = rk4_nsteps(f, y0, t0, t_end, h)


plt.plot(t_values, eyler_values, label='Eyler')
plt.plot(t_values, rk_values, label='Runge-Kutta')
plt.plot(t_values, analytical_values, label='True solution', ls='--')
plt.legend()
plt.savefig('plots/eyler_vs_rk.png', dpi=300)

eyler_error = np.abs(eyler_values - analytical_values).max()
rk_error = np.abs(rk_values - analytical_values).max()
eyler_runge_error = np.abs(runge_error(eyler, f, y0, t0, t_end, h, 1)).max()
rk_runge_error = np.abs(runge_error(rk4_nsteps, f, y0, t0, t_end, h, 4)).max()

print(f'Eyler Absolute error = {eyler_error}')
print(f'RK4 Absolute error = {rk_error}')
print()
print(f'Eyler Runge error = {eyler_runge_error}')
print(f'RK4 Runge error = {rk_runge_error}')
print()


for i in range(8):
    h /= 10
    N = int((t_end - t0) / h)
    t_values = np.linspace(t0, t_end, N + 1)
    
    eyler_values = np.array(eyler(f, y0, t0, t_end, h))
    analytical_values = analytical_solution(t_values)
    
    eyler_error = np.abs(eyler_values - analytical_values).max()
    if rk_error >= eyler_error:
        break
    

print(f'For h = {h} Eyler:')
print(f'N = {N} points')
print(f'Eyler method error = {eyler_error}')
