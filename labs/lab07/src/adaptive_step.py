import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from ode import eyler_adaptive, eyler_modified


@njit
def f(t, y):
    return - 1/3 * y * np.sqrt(t) + 2/3 * (y ** 2) * np.sin(t)


t0, t_end, y0 = 2, 10, 2.2
h0 = 0.05
n = int((t_end - t0) / h0)

ys_eyler_m, ts_eyler_m = eyler_adaptive(f, y0, t0, t_end, h0, 10e-4)
ys_eyler_m = np.array(ys_eyler_m)
ts_eyler_m = np.array(ts_eyler_m)

ys_eyler = eyler_modified(f, y0, t0, t_end, h0)
ts_eyler = np.linspace(t0, t_end, n + 1, endpoint=True)

np.save('eyler_adaptive_values.npy', ys_eyler)
np.save('eyler_adaptive_points.npy', ts_eyler)
hs = ts_eyler[1:] - ts_eyler[:-1]
np.save('eyler_adaptive_steps.npy', hs)

print('Enter option: 1 - table, 0 - plot')
opt = int(input())
if opt:
    for t, y in zip(ts_eyler_m, ys_eyler_m):
        print(f't = {t:.4f}, y = {y:.4f}')
else:
    plt.plot(ts_eyler_m, ys_eyler_m, label='Adaptive Eyler')
    plt.plot(ts_eyler, ys_eyler, label='Eyler')
    plt.legend()
    plt.savefig('plots/adaptive_step.png', dpi=300)
    plt.show()
