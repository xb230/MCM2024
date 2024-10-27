import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from flow_analysis import h0, X1, X2, D, flow_law, flow_mar, get_ont
from hydrological_analysis import Q_s
from scipy.optimize import minimize


deep_cyan = '#30688D'
light_green = '#35B777'

Q = Q_s(0)


def ReLU(x):
    return x if x > - 10 ** 2 else -100


def h_diff(a, b, mode='varied'):
    if mode == 'const' or mode == 'plot':
        return X1() @ np.ones(12) * a + X2() @ np.ones(12) * b + Q + D()
    return X1() @ a + X2() @ b + Q + D()


def h(a1, a2, mode='varied'):
    if mode == 'plot':
        fig, ax1 = plt.subplots()
    h_0 = h0()
    diff = h_diff(a1, a2, mode)
    diff = np.reshape(diff, (12, 4))
    diff = diff.T
    h = np.zeros((4, 12))
    for i in range(4):
        h[i, 0] = h_0[i]
    for i in range(4):
        h[i][0] += diff[i][0]
        for j in range(1, 12):
            h[i][j] = h[i][j - 1] + diff[i][j - 1]
        if mode == 'plot':
            ax1.plot(range(1, 13), h[i])
    return h


def weights():
    r = []
    for i in range(3):
        r.append([0.25, 0.3, 0.1, 0.15, 0.05, 0.05, 0.05, 0.05])
    for i in range(3):
        r.append([0.2, 0.23, 0.17, 0.15, 0.05, 0.05, 0.1, 0.05])
    for i in range(3):
        r.append([0.15, 0.25, 0.1, 0.15, 0.05, 0.1, 0.15, 0.05])
    for i in range(3):
        r.append([0.31, 0.32, 0.12, 0.05, 0.05, 0.05, 0.05, 0.05])
    return np.array(r)


def U_fsh(h1, h2):
    return 1.257 * h1 + 0.945 * np.abs(h1 - h2) + 0.324 + 1.48 * U_bio(h1, h2) - 0.000126 * U_hyd(h1, h2)


def U_bio(h1, h2):
    return 0.133 * h1 + 0.457 * np.abs(h1 - h2) + 0.130 - 0.00119 * U_con(h1, h2)


def U_rec(h1, h2):
    return 0.772 * h1 + 1.065 * (h1 - h2) + 0.986 - 0.0027 * U_con(h1, h2)


def U_spp(h1, h2):
    r = -1.82 * h1 + 1.54 * np.abs(h1 - h2) - 3.54
    return -np.exp(2 * r) if r < 20 else -np.exp(20)


def U_hyd(h1, h2):
    r = 0.824 * (h1 - h2) - 5.46
    return -np.exp(r) if r < 20 else -np.exp(20)


def U_con(h1, h2):
    r = 1.33 * h1 + 1.99 * np.abs(h1 - h2) - 3.816 - 0.000273 * U_hyd(h1, h2)
    return -np.exp(r) if r < 20 else -np.exp(20)


def U_pro(h1, h2):
    r = 0.624 * h1 + 1.177 * np.abs(h1 - h2) + 0.268 - 0.0437 * U_con(h1, h2) - 0.000472 * U_hyd(h1, h2)
    return -np.exp(3 * r) if r < 10 else -np.exp(30)


def U_irr(h1, h2):
    return -2.39 * h1 -12.58 * np.abs(h1 - h2)\
        + 15.93 * U_bio(h1, h2) - 0.00486 * U_hyd(h1, h2) - 0.150 * U_con(h1, h2) + 10.88


def U_vec(h1, h2):
    return np.array([U_fsh(h1, h2), U_bio(h1, h2), U_rec(h1, h2), U_spp(h1, h2), U_hyd(h1, h2), U_con(h1, h2), U_pro(h1, h2), U_irr(h1, h2)])


def U(hs, mode='scalar'):
    if mode =='vector':
        R = []
        for i in range(4):
            r = []
            r.append(U_vec(hs[i, 0], 0) @ weights()[0])
            for j in range(1, 12):
                r.append(U_vec(hs[i, j], hs[i, j - 1]) @ weights()[j])
            R.append(r)
        R_sum = np.sum(np.array(R), axis=0)
        return R_sum
    r = 0
    for i in range(4):
        r += U_vec(hs[i, 0], 0) @ weights()[0]
        for j in range(1, 12):
            r += U_vec(hs[i, j], hs[i, j - 1]) @ weights()[j]
    return r


# X = np.linspace(0.05, 0.11, 100)
# Y = np.linspace(0.7, 1.1, 100)
# Z = np.array([[ReLU(U(h(x, 1, 'const'))) for y in Y] for x in X])
# X, Y = np.meshgrid(X, Y)
#
# ax2 = plt.axes(projection='3d')
# ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)


count = 0

def obj(arguments):
    a1 = arguments[0:12]
    a2 = arguments[12:24]
    global count
    count += 1
    print(count)
    return -U(h(a1, a2))


# x0 = np.concatenate((np.ones(12) * 0.11, np.ones(12) * 1))
# bounds = [(0.11, 0.13)] * 12 + [(0.9, 1.1)] * 12
# result = minimize(obj, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 10**4})
# print(result.fun)
# print(result.x)
# opt_sl = result.x[0:12]
# opt_ms = result.x[12:24]
#
#
# plt.plot(range(1, 13), get_ont(), color=deep_cyan)
# plt.plot(range(1, 13), h(opt_sl, opt_ms, mode='vector')[3] + np.ones(12) * get_ont()[0], color=light_green)
#
# plt.show()