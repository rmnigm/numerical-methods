import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import matplotlib.pyplot as plt

n = 151
a, b = 2.2, 4.2
ua, ub = 0.2, 4
h = (a - b) / n


def delta_f(x0, c):
    return lambda x: c if x == x0 else 0


def k2(k_1, k_2):
    return lambda x: k_1 if x <= (a + b) / 2 else k_2


def k3(k_1, k_2, k_3):
    return lambda x: (k_1 if x <= (a + (b - a) / 3) else
                      (k_3 if x > (a + 2 * (b - a) / 3) else k_2))


options = [
    {'name': '$k_1 << k_2$', 'k': k2(5, 30), 'f': delta_f((a + b) / 2, 10)},
    {'name': '$k_1 >> k_2$', 'k': k2(30, 5), 'f': delta_f((a + b) / 2, 10)},
    {'name': '$k_1 < k_2 < k_3$', 'k': k3(1, 10, 40), 'f': delta_f((a + b) / 2, 10)},
    {'name': '$k_1 = 10k_2 = k_3$', 'k': k3(40, 10, 1), 'f': delta_f((a + b) / 2, 10)},
    {'name': '$100k_1 = k_2 = 100k_3$', 'k': k3(1, 10, 40), 'f': delta_f((a + b) / 2, 10)},
    {'name': 'two symmetrical equal $f(x)$', 'k': k2(5, 30),
        'f': lambda x: delta_f((a + b) / 3, 10)(x) + delta_f(2 * (a + b) / 3, 10)(x)},
    {'name': 'two symmetrical not equal $f(x)$', 'k': k2(5, 30),
        'f': lambda x: delta_f((a + b) / 3, 10)(x) + delta_f(2 * (a + b) / 3, 50)(x)},
    {'name': 'custom $f(x)$', 'k': k2(5, 30),
        'f': lambda x: delta_f((a + b) / 10, 50)(x) + delta_f(2 * (a + b) / 5, 100)(x)},
]
solutions = []

fig, ax = plt.subplots(figsize=(10, 15), nrows=4, ncols=2, sharex=True, sharey=True)
for j, option in enumerate(options):
    name, k, f = option['name'], option['k'], option['f']
    left = np.zeros((n, n))
    right = np.zeros(n)
    x = np.linspace(a, b, n, endpoint=True)
    
    for i in range(1, n - 1):
        left[i][i] = k((x[i] + x[i + 1]) / 2) + k((x[i] + x[i - 1]) / 2)
        left[i][i - 1] = - k((x[i] + x[i - 1]) / 2)
        left[i][i + 1] = - k((x[i] + x[i + 1]) / 2)
        right[i] = f(x[i]) * (h ** 2)
    
    left[0][0] = 1
    left[n - 1][n - 1] = 1
    left = sps.csr_matrix(left)
    right[0] = ua
    right[n - 1] = ub
    u = sps.linalg.spsolve(left, right)
    solutions.append((x, u))
    ax[j // 2][j % 2].plot(x, u, label=name)
    ax[j // 2][j % 2].set_title(name)

plt.tight_layout()
plt.savefig('plots/finite_diff_varying_k.png', dpi=300)
plt.show()
