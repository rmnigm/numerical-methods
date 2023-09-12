import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import njit

from ode import eyler, rk_nsteps


@njit
def f(t, y):
    return - y / t + 3 * t


@njit
def analytical_solution(t):
    return t ** 2


t0, T, y0 = 1, 2, 1
h = 0.1
N = int((T - t0) / h)

t_values = np.array([t0 + i * h for i in range(N + 1)])

analytical_values = np.array([analytical_solution(t) for t in t_values])
eyler_values = np.array(eyler(f, y0, t0, h, N))

_, rk_values = zip(*rk_nsteps(f=f, h=h, t0=t0, y0=y0, t_end=T))


plt.plot(t_values, eyler_values, label='Eyler')
plt.plot(t_values, rk_values, label='Runge-Kutta')
plt.plot(t_values, analytical_values, label='True solution', ls='--')
plt.legend()
plt.savefig('plots/eyler_vs_rk.png', dpi=300)

eyler_error = np.abs(eyler_values - analytical_values).max()
rk_error = np.abs(rk_values - analytical_values).max()

print(f'Eyler method error = {eyler_error}')
print(f'Runge Kutta method error = {rk_error}')
print()


for i in range(8):
    h /= 10
    N = int((T - t0) / h)
    t_values = np.linspace(t0, T, N + 1)
    
    eyler_values = np.array(eyler(f, y0, t0, h, N))
    analytical_values = analytical_solution(t_values)
    
    eyler_error = np.abs(eyler_values - analytical_values).max()
    if rk_error >= eyler_error:
        break
    

print(f'For h = {h} Eyler:')
print(f'N = {N} points')
print(f'Eyler method error = {eyler_error}')
