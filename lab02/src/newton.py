import math

from utils import multiplicity_newton


def ugly_func(x: float) -> float:
    return (32 * math.sqrt(2) * math.sin(x) + 8 * math.pi + 16 * x**2
            + math.pi**2 - 32 - 8 * math.pi * x - 32 * x)


print('m - root multiplicity')
print('x - root')
print('cnt - number of iterations')
for m in range(1, 6):
    x, cnt = multiplicity_newton(ugly_func, (0.5, 1), m, 1e-5)
    print(f'm = {m}: x = {x:.7f}, cnt = {cnt}')
