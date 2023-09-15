import numpy as np
from solve import simple_iterative

A = np.array([
    [79.2, 0, 35, 19.8, 24],
    [39.6, 85, 0, 19.8, 25],
    [19.8, -15, 45, 0, 10],
    [49.5, 18, 20, 89.1, 0],
    [9.9, 15, 20, -49.5, 95],
])
b = np.array([-468.1, 122.3, -257.2, -223.6, 35.9])
n = 5


np_solution = np.linalg.solve(A, b)
print("Numpy solution:")
print(np_solution)

si_solution, iter_cnt = simple_iterative(A, b, max_iter=10)
eps = np.linalg.norm(np_solution - si_solution, ord=np.inf)
print("Simple Iterative solution:")
print(si_solution)
print(f"Precision: {eps:.6f}")
