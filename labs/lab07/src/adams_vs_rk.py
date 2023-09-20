import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from ode import extrapolate_adams, rk4_nsteps, runge_error


@njit
def f(t, y):
    return - (t * y) + (t - 1) * np.exp(t) * (y ** 2)


t0, t_end, y0 = 0, 1, 1
h = 0.1
n = int(2 * (t_end - t0) / h)

t_values = np.array([t0 + i * h / 2 for i in range(n + 1)])[::2]
rk_solution = rk4_nsteps(f, y0, t0, t_end, h / 2)[::2]
rk_errors = runge_error(rk4_nsteps, f, y0, t0, t_end, h, 4)

y0_adams = rk_solution[:2]
t0_adams = t_values[:2]

adams_solution = extrapolate_adams(f, y0_adams, t0_adams, t_end, h / 2)[::2]
adams_errors = runge_error(extrapolate_adams, f, y0_adams, t0_adams, t_end, h, 2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].plot(t_values, rk_solution, label='Base RK4', color='green')
ax[0].plot(t_values, rk_solution + rk_errors, label='Precise RK4', color='blue')
ax[1].plot(t_values, adams_solution, label='Base Adams')
ax[1].plot(t_values, adams_solution + adams_errors, label='Precise Adams')
fig.legend()
plt.savefig('plots/adams_vs_rk.png', dpi=300)
