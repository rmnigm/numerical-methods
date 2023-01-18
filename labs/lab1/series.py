import matplotlib.pyplot as plt


def calculate_nth(n):
    return 32 / (n ** 2 + 5 * n + 6)


sums = {}
for n in (10 ** p for p in range(5)):
    s = 0.
    for i in range(n):
        s += calculate_nth(i)
    sums[n] = s

keys = list(map(str, sums.keys()))
errors = list(map(lambda x: 16 - x, sums.values()))

bars = plt.barh(keys, errors)
plt.bar_label(bars, padding=8, fontsize=9)
plt.xlim(0, 14)
plt.savefig('series.png', dpi=300)
