#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_nth(n):
    return 32 / (n ** 2 + 5 * n + 6)


sums = {}
errors = {}
n_digits = defaultdict(int)
for N in (10 ** p for p in range(5)):
    s = 0.
    true_s = 16.
    for i in range(N):
        s += calculate_nth(i)
    sums[N] = s
    errors[N] = true_s - s
    i = -1
    while i < 30:
        if errors[N] <= 10 ** (-i):
            n_digits[N] += 1
            i += 1
        else:
            break
    print(f'Sum, Error, Number of digits for n = {N}')
    print(sums[N], errors[N], n_digits[N])

keys = list(map(str, sums.keys()))
errors = list(errors.values())
n_digits = list(n_digits.values())

bars = plt.barh(keys, errors, label='Погрешность')
plt.bar_label(bars, padding=8, fontsize=9)
plt.xlim(0, 14)
plt.legend()
plt.savefig('plots/series_error.png', dpi=300)
plt.close()

bars = plt.barh(keys, n_digits, label='Кол-во значимых цифр')
plt.bar_label(bars, padding=8, fontsize=9)
plt.xlim(0, 5)
plt.legend()
plt.savefig('plots/series_n_digits.png', dpi=300)
