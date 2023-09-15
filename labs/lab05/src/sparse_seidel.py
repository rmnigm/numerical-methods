import numpy as np

from sparse_matrix import SparseMatrix
from solve import seidel


A = SparseMatrix(n=50)
b = np.zeros(50)
for i in range(50):
    A[i, i] = 100
    j = i + 1
    b[i] = j * np.exp(10 / j) * np.cos(9 / j)
for i in range(50 - 1):
    A[i, i + 1] = 27
for i in range(50 - 3):
    A[i, i + 3] = 15
for i in range(50 - 7):
    A[i, i + 7] = 1

np_solution = np.linalg.solve(A.dense(), b)
seidel_solution, iter_cnt = seidel(A, b, eps=1e-9)
eps = np.linalg.norm(np_solution - seidel_solution, ord=np.inf)

print("Numpy solution:")
print(np_solution)

print("\nSeidel method solution")
print(seidel_solution)
print(f"Iterations count = {iter_cnt}")
print(f"Solution precision by infinity norm: {eps:.10f}")