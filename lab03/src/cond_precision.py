import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


N, n = 16, 5
C, b = np.zeros((n, n), dtype=float), np.full(n, fill_value=N, dtype=float)

for i in range(n):
    for j in range(n):
        C[i, j] = 0.1 * N * (i + 1) * (j + 1)
        
A = 100 / (3 + 0.3 * C) ** 5

x = np.linalg.solve(A, b)

cond_value = np.max(np.abs(A))
delta = 0.1

x_modified = np.empty((n, n))
for i in range(n):
    b_modified = b.copy()
    b_modified[i] += delta
    x_modified[i] = np.linalg.solve(A, b_modified)
d = np.array([np.max(x - x_i) / np.max(x) for x_i in x_modified])

plt.figure(figsize=(6, 5))
plt.bar(range(n), d)
plt.xlabel('')
plt.savefig('plots/cond_precision.png', dpi=300)

d_argmax = np.argmax(d)
b_modified = b.copy()
b_modified[d_argmax] += delta


with np.printoptions(precision=5):
    rel_delta = np.max(b_modified - b) / np.max(b)
    print(f'm = {d_argmax + 1}')
    print(f'd = {d}')
    print(f'delta(x^m) = {d[d_argmax]}')
    print(f'delta(b^m) = {rel_delta}')
    print(f'cond(A) = {cond_value}')
    cmp_sign = '<=' if d[d_argmax] <= rel_delta * cond_value else '>'
    print(f'{d[d_argmax]} {cmp_sign} {rel_delta * cond_value}')
    print(f'delta(x^m) {cmp_sign} cond(A) * delta(b^m)')
