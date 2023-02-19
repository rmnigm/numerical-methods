import math
import typing as tp

import scipy
import numpy as np
import matplotlib.pyplot as plt

from utils import bisection


def root_finder(func: tp.Callable, name: str, interval: tuple[float, float]) -> None:
    print(f'{name}(x):')
    scipy_root = scipy.optimize.root(fun=func, x0=-1).x[0]
    bisect_root = bisection(func, 1e-10, interval)
    print(f'scipy root = {scipy_root:.5f}')
    print(f'bisection = {bisect_root:.5f}')
    
    x = np.linspace(-1, 0, 100)
    vals = np.vectorize(func)(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, vals)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xticks(np.arange(-1, 0.1, 0.1))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(f'$y={name}(x)$ graph near root')
    plt.savefig(f'plots/bisec_{name}.png', dpi=300)


functions = {
    'f': lambda x: math.sin(x)**2 + 5/6 * math.sin(x) + 1/6,
    'g': lambda x: math.sin(x)**2 + 2/3 * math.sin(x) + 1/9,
}

for key, func in functions.items():
    root_finder(func, key, (-1, 0))
