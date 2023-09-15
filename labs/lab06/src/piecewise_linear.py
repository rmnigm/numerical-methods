import numpy as np
import matplotlib.pyplot as plt


def linear_interpolate(x0, y0, x1, y1):
    return lambda x: (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)


def poly_newton_coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a


def newton_polynomial(x_data, y_data, x):
    a = poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k]) * p
    return p


left, right = 0, 2
k = 10
step = (right - left) / k
x, wide_x = [], []
for i in range(1, k + 1):
    val = i * step
    x.append(val)
    wide_x.append(val - 2 * step / 3)
    wide_x.append(val - step / 3)
    wide_x.append(val)
x, wide_x = np.array([0.] + x), np.array([0.] + wide_x)

y = (x + 1) * np.abs(x ** 2 - 2)
wide_y = (wide_x + 1) * np.abs(wide_x ** 2 - 2)
linear_inter = []
newton_inter = np.array(newton_polynomial(x, y, wide_x))

j = 0
for i in range(k):
    f = linear_interpolate(x[i], y[i], x[i + 1], y[i + 1])
    while j < 3 * k + 1 and wide_x[j] <= x[i + 1]:
        linear_inter.append(f(wide_x[j]))
        j += 1

linear_inter = np.array(linear_inter)

plt.plot(wide_x, wide_y, label='actual $f$')
plt.plot(wide_x, linear_inter, label='piecewise $f$')
plt.plot(wide_x, newton_inter, label='newton $f$')
plt.scatter(x, y, label='data')
plt.legend()
plt.savefig('plots/piecewise_linear_interpolation.png', dpi=400)
plt.close()

plt.plot(wide_x, np.abs(linear_inter - wide_y), label='$\Delta f$ piecewise')
plt.plot(wide_x, np.abs(newton_inter - wide_y), label='$\Delta f$ newton')
plt.legend()
plt.savefig('plots/piecewise_linear_errors.png', dpi=400)
plt.close()
