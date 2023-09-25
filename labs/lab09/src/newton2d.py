import numpy as np
import matplotlib.pyplot as plt

from optimize import newton_minimize_vec


def f(x):
    return x[0] ** 2 + 2 * x[1] ** 2 - 4 * np.sin(x[0]) - np.sin(x[1])


def ff(x, y):
    return x ** 2 + 2 * y ** 2 - 4 * np.sin(x) - np.sin(y)


bbox = (np.array([-2, -2]), np.array([4, 4]))

nx, ny = (50, 50)
x = np.linspace(bbox[0][0], bbox[1][0], nx)
y = np.linspace(bbox[0][1], bbox[1][1], ny)
xv, yv = np.meshgrid(x, y)
zv = ff(xv, yv)

plt.subplots(figsize=(8, 5))
plt.contourf(x, y, zv)
plt.axis('scaled')
plt.colorbar()
plt.savefig('pics/newton2d_levels.png', dpi=300)

print('Enter starting points in format < x y >:')
start = np.array(list(map(float, input().split())))

minima, iter_min = newton_minimize_vec(func=f,
                                       bbox=bbox,
                                       start=start,
                                       eps=1e-6,
                                       minimize=True)
maxima, iter_max = newton_minimize_vec(func=f,
                                       bbox=bbox,
                                       start=start,
                                       eps=1e-6,
                                       minimize=False)


fig, ax = plt.subplots(figsize=(7, 6))
plt.contour(x, y, zv)
plt.axis('scaled')
plt.colorbar()
plt.scatter(minima[0], minima[1],
            s=70,
            color='red',
            label=f'$min$ $f(x)$, {iter_min} iterations')
plt.scatter(maxima[0], maxima[1],
            s=70,
            color='green',
            label=f'$max$ $f(x)$, {iter_max} iterations')
plt.tight_layout()
plt.legend()
plt.savefig('pics/newton2d.png', dpi=300)
