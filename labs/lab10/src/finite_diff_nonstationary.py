import imageio.v2 as imageio
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tqdm import trange


n = 10
f = lambda x:  10 * (x ** -1/4)
k = lambda x: x ** 3
a, b = 0.1, 1.1
tau = 0.05
T0, T = 0, tau * 100
ua, ub = 3, 0
h = 0.1

x = np.linspace(a, b, n, endpoint=True)


def iteration(t):
    left = np.zeros((n, n))
    right = np.zeros(n)
    for i in range(1, n - 1):
        k_right = k((x[i] + x[i + 1]) / 2)
        k_left = k((x[i] + x[i - 1]) / 2)
        left[i][i] = k_right + k_left
        left[i][i - 1] = - k_left - (h / 2)
        left[i][i + 1] = - k_right + (h / 2)
        right[i] = f(x[i]) * (1 - np.exp(-t)) * (h ** 2)
    
    left[0][0] = 1
    right[0] = ua
    left[n - 1][n - 1] = 1
    right[n - 1] = ub
    
    left = sps.csr_matrix(left)
    u = sps.linalg.spsolve(left, right)
    return u


max_iter = 100
step = 1
for i in trange(0, max_iter, step):
    plt.figure(figsize=(10, 5))
    u = iteration(i * tau)
    color = (1 - i / max_iter, i / max_iter, 0)
    plt.plot(x, u, color=color)
    plt.savefig(f'plots/gif/{i}.png')
    plt.close()

filenames = [f'plots/gif/{i}.png' for i in range(0, max_iter, step)]

with imageio.get_writer('plots/finite_difference_method.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
