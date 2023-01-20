import numpy as np

coefficients = np.array([1, -30.9, 238.7])
errors = (10**p for p in range(-15, -1))

for error in errors:
    actual_coefficients = coefficients + np.array([0, 0, error])
    print('Error:', error)
    print('Roots:', np.roots(actual_coefficients))
