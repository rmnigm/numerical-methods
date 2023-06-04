import numpy as np
import matplotlib.pyplot as plt


m = 3
y_target = 5
x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
y = np.array([3.69, 3.9, 4.3, 4.97, 5.96, 7.35, 9.2, 11.57, 14.54])
X = np.stack([x ** k for k in [0, m]]).T
weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
k, b = weights
print(f'f = {k:.5f} * x + {b:.5f}')

x_net = np.arange(0, 3, 0.01)
X_net = np.stack([x_net ** k for k in [0, m]]).T
y_net = X_net.dot(weights)
plt.plot(x_net, y_net, label='approximated $f$')
plt.scatter(x, y, label='data points')
plt.legend()
plt.savefig('plots/material_point.png', dpi=400)

x_target = ((y_target - b) / k) ** (1 / m)
print(x_target)
