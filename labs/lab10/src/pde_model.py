import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import typing as tp


x = sp.Symbol('x')
c, c1, c2 = sp.symbols('c c1 c2')
f = 10 * (x ** -1/4)

a, b = 0.1, 1.1
base_k = x ** 3
ua, ub = 3, 0

k_options = [base_k, c * base_k, c * base_k, 1 / base_k, base_k, base_k, base_k]
bounds_values_options = [
    (ua, ub), (ua, ub), (ua, ub), (ua, ub),
    (-ua, ub), (ua, -ub), (-ua, -ub)
]
c_options = [1, 10, 0.1, 1, 1, 1, 1]

options = list(zip(k_options, bounds_values_options, c_options))


def symbol_integrate(f, x, k, c1, c2, conditions):
    num = sp.integrate(-f, x)
    u = sp.integrate(num / k, x) + sp.integrate(c1 / k, x) + c2
    system = [u.subs(condition) - condition['u'] for condition in conditions]
    return u.subs(sp.solve(system, c1, c2)).subs(c, C)


ps = [None, None, None]
subsets = [[0, 1, 2], [0, 3], [4, 5, 6]]

for p, subset in zip(ps, subsets):
    for i in subset:
        k, (ua, ub), C = options[i]
        conditions = [{'c': C, 'x': a, 'u': ua}, {'c': C, 'x': b, 'u': ub}]
        u = symbol_integrate(f, x, k, c1, c2, conditions)
        label = f'$c={C:.1f}, UA={ua:.1f}, UB={ub:.1f}, k={sp.latex(k)}$'
        p_temp = sp.plotting.plot(u, (x, a, b),
                                  axis_center=(0, 0),
                                  xlim=(0, b),
                                  show=False,
                                  label=label,
                                  legend=True)
        if p is None:
            p = p_temp
        else:
            p.extend(p_temp)
    name = '_'.join([str(i + 1) for i in subset])
    p.save(f'plots/pde_model_{name}.png')
