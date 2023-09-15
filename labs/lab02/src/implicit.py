import math
import typing as tp

import matplotlib.pyplot as plt  # type: ignore
from scipy.optimize import root  # type: ignore


def implicit_to_normal(x_stable: float):
    def decorator(func: tp.Callable):
        def wrapper(y):
            return func(x_stable, y)
        return wrapper
    return decorator


x_interval = (1, 5)
y_interval = (0.1, 1.2)
x_vals = [x / 2 for x in range(2, 11)]
y_vals = []


def calc_point(x_val: float):
    
    @implicit_to_normal(x_stable=x_val)
    def implicit_function(x, y):
        return (math.sinh(y * math.exp(y) - x / 20)
                + math.atan(20 * y * math.exp(y) - x) - 0.5)
    
    return root(fun=implicit_function, x0=y_interval[1])


for x in x_vals:
    sol = calc_point(x)
    y_vals.append(sol.x[0])
    
for x, y in zip(x_vals, y_vals):
    print(f'x={x}: y={y:.7f}')

plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals)
plt.xticks(x_vals)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$y=f(x)$ from implicit function $F(x, y) = 0$')
plt.savefig('plots/implicit.png', dpi=300)
