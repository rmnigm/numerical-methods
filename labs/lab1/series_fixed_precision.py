import matplotlib.pyplot as plt

from math import ceil
from collections import defaultdict


def fixed_bit_depth(x, m):
    p = 0
    while x >= 1:
        x /= 10
        p += 1
    x *= 10**p
    return ceil(x * 10**(m - p)) / 10**(m - p)


def calculate_nth(n, m):
    true_val = 32 / (n ** 2 + 5 * n + 6)
    return fixed_bit_depth(true_val, m)


true_s = 16.
N = 10000
sums, errors, n_digits = {}, {}, defaultdict(int)

for bit_depth in range(4, 9):
    s = 0.
    for i in range(N):
        s += calculate_nth(i, bit_depth)
    sums[bit_depth] = s
    errors[bit_depth] = true_s - s
    i = -1
    while i < 30:
        if abs(errors[bit_depth]) <= 10 ** (-i):
            n_digits[bit_depth] += 1
            i += 1
        else:
            break
    print(f'Sum, Error, Number of digits for bit_depth = {bit_depth}')
    print(sums[bit_depth], errors[bit_depth], n_digits[bit_depth])

keys = list(map(str, sums.keys()))
errors = list(errors.values())
n_digits = list(n_digits.values())

bars = plt.barh(keys, errors, label='Погрешность')
plt.bar_label(bars, padding=8, fontsize=9)
plt.xlim(-1.2, 0)
plt.legend()
plt.savefig('plots/series_fixed_error.png', dpi=300)
plt.close()

bars = plt.barh(keys, n_digits, label='Кол-во значимых цифр')
plt.bar_label(bars, padding=8, fontsize=9)
plt.xlim(0, 5)
plt.legend()
plt.savefig('plots/series_fixed_n_digits.png', dpi=300)
