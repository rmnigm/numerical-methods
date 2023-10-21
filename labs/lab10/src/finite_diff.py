import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib.pyplot as plt

f = lambda x: -3
q = lambda x: 0.5

solutions = []
base_n = 100
for i, n in enumerate([base_n, 2 * base_n]):
    a, b = 2.2, 4.2
    ua, ub = 0.2, 4
    h = (a - b) / n
    
    left = np.zeros((n, n), dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    x = np.linspace(a, b, n, endpoint=True)
    
    for i in range(1, n - 1):
        left[i][i] = 2 + (h ** 2) * q(x[i])
        left[i][i - 1] = -1 - (1 / 2) * h
        left[i][i + 1] = -1 + (1 / 2) * h
        right[i] = f(x[i]) * (h ** 2)
    
    left[0][0] = 1 - 3 / h
    left[0][1] = 4 / h
    left[0][2] = -1 / h
    left[n - 1][n - 1] = 3 / h
    left[n - 1][n - 2] = -4 / h
    left[n - 1][n - 3] = -2 / h
    left = sps.csr_matrix(left)
    right[0] = ua
    right[n - 1] = ub
    u = sps.linalg.spsolve(left, right)
    solutions.append((x, u, left, right))

errors = (solutions[1][1][::2] - solutions[0][1]) / 3
max_error = np.abs(errors).max()
print(f'Error = {max_error:.4f} {"<" if max_error < 0.03 else ">"} {3e-2}')
x, u, left, right = solutions[0]
plt.plot(x, u, label='Base solution')
plt.plot(x + errors, u, label='Precise solution')
plt.legend()
plt.savefig('plots/finite_diff.png', dpi=300)
