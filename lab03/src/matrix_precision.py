import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


N, n = 16, 5
C, b = np.zeros((n, n), dtype=float), np.full(n, fill_value=N, dtype=float)

for i in range(n):
    for j in range(n):
        C[i, j] = 0.1 * N * (i + 1) * (j + 1)
        
A = 100 / (3 + 0.3 * C) ** 5
x = np.linalg.solve(A, b)
cond_value = np.max(A)

delta = 0.1
x_modified = {}
for i in range(n):
    for j in range(n):
        A_modified = A.copy()
        A_modified[i, j] += delta
        x_modified[(i, j)] = np.linalg.solve(A_modified, b)
d = {key: np.max(x - x_i) / np.max(x) for key, x_i in x_modified.items()}

plt.figure(figsize=(10, 5))
plt.bar([str(el) for el in d.keys()], d.values())
plt.xlabel('')
plt.xticks(rotation=90)
plt.savefig('plots/matrix_precision.png', dpi=300)

d_i, d_j = max(d, key=d.get)  # type: ignore
A_modified = A.copy()
A_modified[d_i, d_j] += delta
rel_delta = np.max(A_modified - A) / np.max(A)
cmp_sign = '<=' if d[(d_i, d_j)] <= rel_delta * cond_value else '>'

print(f'i, j = {d_i, d_j}')
print(f'delta(x^h) = {d[(d_i, d_j)]}')
print(f'delta(A^h) = {rel_delta}')
print(f'cond(A) = {cond_value}')
print(f'{d[(d_i, d_j)]} {cmp_sign} {rel_delta * cond_value}')
print(f'delta(x^h) {cmp_sign} cond(A) * delta(A^h)')
