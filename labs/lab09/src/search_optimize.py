import numpy as np
import matplotlib.pyplot as plt

from optimize import fibonacci


def f(t):
    return (t ** 2 - 3) / (t ** 2 + 2)


interval = (-1, 4)

minima, iter_min = fibonacci(func=f, interval=interval, eps=1e-6)
maxima, iter_max = fibonacci(func=f, interval=interval, eps=1e-6, minimize=False)
t = np.linspace(start=interval[0], stop=interval[1], num=100)
y = f(t)

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(t, y, label='$f(x)$')
plt.scatter(minima, f(minima),
            s=70, color='red', label=f'$min$ $f(x)$, {iter_min} iterations'
            )
plt.scatter(maxima, f(maxima),
            s=70, color='green', label=f'$max$ $f(x)$, {iter_max} iterations'
            )
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('plots/search_optimize.png', dpi=300)
