import numpy as np
import matplotlib.pyplot as plt
from optimize import minimize


def distance_f(p, a1, a2, a3):
    def f(phi):
        z = np.zeros(3, dtype=np.double)
        z[0] = a1 * np.sin(phi[0]) * np.sin(phi[1])
        z[1] = a2 * np.sin(phi[0]) * np.cos(phi[1])
        z[2] = a3 * np.cos(phi[0])
        return np.sum((z - p)**2)
    return f


def inv_transform(a1, a2, a3):
    def inv(phi):
        return np.array([
            a1 * np.sin(phi[0]) * np.sin(phi[1]),
            a2 * np.sin(phi[0]) * np.cos(phi[1]),
            a3 * np.cos(phi[0])
        ])
    return inv


ps = np.array([
    [14, 8.2, 13.011],
    [7.425, 7.532, 9.758],
    [13.125, 4.438, 5.75]
])
N = 16
a1 = 8.5 - N * 0.25
a2 = 2.3 + N * 0.3
a3 = 4 - N * 0.1
distances = {}

for p in ps:
    solution = None
    res_dist = np.inf
    for i in np.arange(0, np.pi, 0.1):
        angles = np.array([i, i], dtype=np.float16)
        inv = inv_transform(a1, a2, a3)
        dist = distance_f(p, a1, a2, a3)
        solution_angles, iter_cnt = minimize(dist, angles)
        if dist(solution_angles) < res_dist:
            res_dist = dist(solution_angles)
            solution = inv(solution_angles)
    distances[tuple(p)] = (solution, res_dist)


with np.printoptions(precision=4):
    for point, (solution, res_dist) in distances.items():
        print(f'Point {point}:')
        print(f'Distance = {res_dist:.4f}, closest point = {solution}')


fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(projection='3d')

coefs = (a1, a2, a3)
rx, ry, rz = 1 / np.sqrt(coefs)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='pink')
ax.scatter(ps.T[0], ps.T[1], ps.T[2], marker='^')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-2, 16))
plt.savefig('plots/point_distances.png', dpi=300)
plt.show()
