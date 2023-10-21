import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt

n = 2000
a, c, b = 0, 1.215, 1.8
ua, ub = 0, 0
h = (a - b) / n


def f(x):
    return 8 * x * (2 - x)


def k(x):
    return 0.4 if x <= c else 1.2


def q(x):
    return 3.2 if x <= c else 8.5


fig, ax = plt.subplots(figsize=(10, 5))
left = np.zeros((n, n), dtype=np.float64)
right = np.zeros(n, dtype=np.float64)
x = np.linspace(a, b, n, endpoint=True)

for i in range(1, n - 1):
    k_right = k((x[i] + x[i + 1]) / 2)
    k_left = k((x[i] + x[i - 1]) / 2)
    q_val = q(x[i])
    left[i][i] = k_right + k_left + (h ** 2) * q_val
    left[i][i - 1] = - k_left
    left[i][i + 1] = - k_right
    right[i] = f(x[i]) * (h ** 2)

k0 = k((x[1] + x[0]) / 2)
f0 = 2 * quad(f, x[0], x[1])[0] / h
q0 = 2 * quad(q, x[0], x[1])[0] / h
left[0][0] = k0 + (h ** 2) / 2 * q0
left[0][1] = -k0
right[0] = ua + f0 * (h ** 2) / 2

k_end = k((x[n - 1] + x[n - 2]) / 2)
f_end = 2 * quad(f, x[n - 2], x[n - 1])[0] / h
q_end = 2 * quad(q, x[n - 2], x[n - 1])[0] / h
left[n - 1][n - 1] = k_end + (h ** 2) / 2 * q_end
left[n - 1][n - 2] = - k_end
right[n - 1] = ub + f_end * (h ** 2) / 2

left = sps.csr_matrix(left)
u = sps.linalg.spsolve(left, right)
plt.plot(x, u)
plt.suptitle('Краевая задача')
plt.tight_layout()
plt.savefig('plots/finite_diff_stream.png', dpi=300)
plt.show()

n *= 2
h = (a - b) / n
left = np.zeros((n, n), dtype=np.float64)
right = np.zeros(n, dtype=np.float64)
x = np.linspace(a, b, n, endpoint=True)

for i in range(1, n - 1):
    k_right = k((x[i] + x[i + 1]) / 2)
    k_left = k((x[i] + x[i - 1]) / 2)
    q_val = q(x[i])
    left[i][i] = k_right + k_left + (h ** 2) * q_val
    left[i][i - 1] = - k_left
    left[i][i + 1] = - k_right
    right[i] = f(x[i]) * (h ** 2)

k0 = k((x[1] + x[0]) / 2)
f0 = 2 * quad(f, x[0], x[1])[0] / h
q0 = 2 * quad(q, x[0], x[1])[0] / h
left[0][0] = k0 + (h ** 2) / 2 * q0
left[0][1] = -k0
right[0] = ua + f0 * (h ** 2) / 2

k_end = k((x[n - 1] + x[n - 2]) / 2)
f_end = 2 * quad(f, x[n - 2], x[n - 1])[0] / h
q_end = 2 * quad(q, x[n - 2], x[n - 1])[0] / h
left[n - 1][n - 1] = k_end + (h ** 2) / 2 * q_end
left[n - 1][n - 2] = - k_end
right[n - 1] = ub + f_end * (h ** 2) / 2

left = sps.csr_matrix(left)
u_second = sps.linalg.spsolve(left, right)

errors = (u_second[::2] - u) / 3
max_error = np.abs(errors).max()
print(f'Error = {max_error:.4f} {"<" if max_error < 0.001 else ">"} 0.001')
