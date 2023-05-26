import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt


def approximate(x, y, m):
    weights, sigma = {}, {}
    for m in range(1, m):
        X = np.stack([x ** k for k in range(m)]).T
        weight, residuals, _, _ = numpy.linalg.lstsq(X, y, rcond=None)
        sigma[m] = np.sqrt((1 / (n - m)) * residuals)
        weights[m] = weight
    plt.plot(sigma.keys(), sigma.values())
    plt.show()
    print('Enter optimal M:')
    optimal = int(input())
    return weights[m], optimal


def plot(weights, m, x, y):
    xs = np.arange(min(x) - 0.1, max(x) + 0.1, 0.01)
    X = np.stack([xs ** k for k in range(m)]).T
    Y = X.dot(weights)
    plt.plot(xs, Y, label='polynomial interpolation')
    plt.scatter(x, y, label='data points')
    plt.legend()
    plt.show()


x = np.array([-3.2, -2.66, -2.12, -1.58, -1.04, -0.5, 0.04, 0.58, 1.12, 1.66, 2.2])
y = np.array([-0.173, -0.574, -1.811, -1.849, 0.123, 1.462, 2.399, 1.3, 1.703, -2.045, 2.817])
n = len(x)

optimal_weights, optimal_m = approximate(x, y, 11)
plot(optimal_weights, optimal_m, x, y)
