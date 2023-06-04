import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt

left, right = 1, 1.28
fig = plt.figure()
ks = [2, 3, 4, 5]  # 6, 8, 10, 12
for i in range(4):
    k = ks[i]
    x = np.arange(left, right, (right - left) / k)
    wide_x = np.arange(left, right, (right - left) / (3 * k))
    y = 12 * np.sin(np.exp(x))
    wide_y = 12 * np.sin(np.exp(wide_x))
    
    poly = intp.CubicSpline(x, y)
    
    plt.plot(wide_x, np.abs(wide_y - poly(wide_x)), label=f'{k}')
    

plt.legend()
plt.suptitle('Error of splines by knot quantity')
plt.show()
plt.savefig('plots/spline_by_knot_error_1.png', dpi=400)
plt.close()
