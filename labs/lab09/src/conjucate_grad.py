import numpy as np
import matplotlib.pyplot as plt

from optimize import conjucate_grad_minimize as conjgrad
from optimize import conjucate_grad_quadratic_minimize as conjgrad_quadr


def get_quad(A, b):
    
    def f_arr(x):
        return x @ A @ x + b @ x
    
    def f_var(x, y):
        return (A[0][0] * x ** 2
                + 2 * A[0][1] * x * y
                + A[1][1] * y ** 2
                + b[0] * x
                + b[1] * y)
    
    return f_arr, f_var


A = np.array([
    [0.5, -0.25],
    [-0.25, 2.5]
])
b = np.array([-2.5, -3.5])
start = np.array([0, 0])
f, ff = get_quad(A, b)

nx, ny = (50, 50)
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
xv, yv = np.meshgrid(x, y)
zv = ff(xv, yv)

minima, iter_min = conjgrad(func=f, start=start, eps=1e-6)
minima_q, iter_min_q = conjgrad_quadr(func=f, func_matrix=A, start=start, eps=1e-6)
print(f'min(f(x)) = {minima}, {iter_min} iterations with general formula')
print(f'min(f(x)) = {minima_q}, {iter_min_q} iterations with quadratic formula')

plt.subplots(figsize=(8, 6))
plt.contourf(x, y, zv, cmap='gray')
plt.scatter(minima[0], minima[1],
            s=70,
            color='red',
            label=f'$min$ $f(x)$, {iter_min} iterations')
plt.axis('scaled')
plt.colorbar()
plt.tight_layout()
plt.legend()
plt.savefig('plots/conjucate_grad.png', dpi=300)
