from math import ceil


def fixed_bit_depth(x, m):
    p = 0
    while x >= 1:
        x /= 10
        p += 1
    x *= 10**p
    return ceil(x * 10**(m - p)) / 10**(m - p)
