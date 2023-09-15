import numpy as np  # type: ignore


def tridiagonal_solve(a: np.array,
                      b: np.array,
                      c: np.array,
                      d: np.array) -> np.array:
    for it in range(1, len(d)):
        mc = a[it - 1] / b[it - 1]
        b[it] = b[it] - mc * c[it - 1]
        d[it] = d[it] - mc * d[it - 1]
    xc = b
    xc[-1] = d[-1] / b[-1]
    for il in range(len(d) - 2, -1, -1):
        xc[il] = (d[il] - c[il] * xc[il + 1]) / b[il]
    return xc


# for matrix from task itself
n = 50
A = np.full(n, fill_value=2, dtype=float)
A[0] = 0
C = np.full(n, fill_value=1, dtype=float)
C[n-1] = 0
B = np.full(n, fill_value=100, dtype=float)
D = (np.arange(1, n+1) * np.exp(10 / np.arange(1, n+1))
     * np.cos(9 / np.arange(1, n+1)))

print('Main diagonal B:')
print(B)
print('Lower subdiagonal A:')
print(A[1:])
print('Upper subdiagonal C:')
print(C[:-1])
print('Coefficients D:')
print(D)
print('Solution X:')
print(tridiagonal_solve(A, B, C, D))

# for smaller matrix, example and proof
n = 3
A = np.full(n, fill_value=4, dtype=float)
A[0] = 0
C = np.full(n, fill_value=3, dtype=float)
C[n-1] = 0
B = np.full(n, fill_value=5, dtype=float)
D = np.arange(1, n+1)
print('Solution for smaller matrix')
print('Main diagonal B:')
print(B)
print('Lower subdiagonal A:')
print(A[1:])
print('Upper subdiagonal C:')
print(C[:-1])
print('Coefficients D:')
print(D)
print('Solution X:')
print(tridiagonal_solve(A, B, C, D))
