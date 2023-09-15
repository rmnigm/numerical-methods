import numpy as np
import matplotlib.pyplot as plt


def lstsq(X, Y, m):
    pows = [0, m]
    b = np.zeros(2)
    G = np.zeros((2, 2))
    for j in range(2):
        b[j] = sum(y * x ** pows[j] for y, x in zip(Y, X))
        for k in range(2):
            G[j, k] = sum(x ** (pows[k] + pows[j]) for x in X)
    return np.linalg.solve(G, b)


m = 3
y_target = 5
x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
y = np.array([3.69, 3.9, 4.3, 4.97, 5.96, 7.35, 9.2, 11.57, 14.54])
X = np.stack([x ** k for k in [0, m]]).T
weights = lstsq(x, y, m)
k, b = weights
print(f'f = {k:.5f} * x ^ {m} + {b:.5f}')

x_net = np.arange(0, 3, 0.01)
X_net = np.stack([x_net ** k for k in [0, m]]).T
y_net = X_net.dot(weights)
plt.plot(x_net, y_net, label='approximated $f$')
plt.scatter(x, y, label='data points')
plt.legend()
plt.show()

x_target = ((y_target - b) / k) ** (1 / m)
print(f'x = {x_target}')
