import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt

left, right = 1, 1.28
fig1, ax1 = plt.subplots(nrows=1, ncols=1)
fig2, ax2 = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))
ks = [3, 4, 6, 8, 10, 12]
for i in range(6):
    k = ks[i]
    x = np.linspace(left, right, k, endpoint=True)
    wide_x = np.linspace(left, right, 3 * k, endpoint=True)
    y = 12 * np.sin(np.exp(x))
    wide_y = 12 * np.sin(np.exp(wide_x))
    
    poly = intp.interp1d(x, y, kind='quadratic')
    
    ax1.plot(wide_x, np.abs(wide_y - poly(wide_x)), label=f'{k}')
    ax2[i // 2][i % 2].plot(wide_x, wide_y, label='f(x)')
    ax2[i // 2][i % 2].plot(wide_x, poly(wide_x), label=f'{k} knots')
    ax2[i // 2][i % 2].legend()
    

fig1.legend()
fig1.suptitle('Error of splines by knot quantity')
fig1.savefig('plots/spline_by_knot_error.png', dpi=400)
plt.close(fig1)


fig2.suptitle('Splines by knot quantity')
fig2.tight_layout()
fig2.savefig('plots/spline_by_knot.png', dpi=400)
plt.close(fig2)
