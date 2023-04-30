import numpy as np
from solve import seidel

A = np.array([
    [79.2, 0, 35, 19.8, 24],
    [39.6, 85, 0, 19.8, 25],
    [19.8, -15, 45, 0, 10],
    [49.5, 18, 20, 89.1, 0],
    [9.9, 15, 20, -49.5, 95],
])
b = np.array([-468.1, 122.3, -257.2, -223.6, 35.9])
n = 5

L = np.tril(A, -1)
U = np.triu(A, 1)
D = np.diag(np.diag(A))
B = np.linalg.inv(L + D).dot(-U)

np_solution = np.linalg.solve(A, b)
print(f"Numpy solution: {np_solution}\n")

converge = np.linalg.norm(B, ord=np.inf) < 1
print(f"System convergence with Seidel iterative method: {converge}")
if converge:
    seidel_solution_1, _ = seidel(A, b, x=np.zeros(n), max_iter=10)
    seidel_solution_2, _ = seidel(A, b, x=np.ones(n), max_iter=10)
    eps_1 = np.linalg.norm(np_solution - seidel_solution_1, ord=np.inf)
    eps_2 = np.linalg.norm(np_solution - seidel_solution_2, ord=np.inf)
    
    print(f"x = {np.zeros(n)}")
    print("Seidel method solution:")
    print(seidel_solution_1)
    print(f"Solution precision by infinity norm: {eps_1:.6f}")
    print(f"x = {np.ones(n)}")
    print("Seidel method solution:")
    print(seidel_solution_2)
    print(f"Solution precision by infinity norm: {eps_2:.6f}")
