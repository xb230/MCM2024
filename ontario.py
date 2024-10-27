import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from flow_analysis import h0, X1, X2, D, flow_law, flow_mar, get_ont
from hydrological_analysis import Q_s
from gradient_descent import U_vec
from scipy.optimize import minimize

deep_cyan = '#30688D'
light_green = '#35B777'


Q = Q_s(0).reshape((12, 4)).T[3]
D_ont = D().reshape((12, 4)).T[3]
print(Q)
print(D_ont)


def weights():
    r = []
    for i in range(3):
        a = [0, 0.3, 0.1, 0.15, 0.05, 0.05, 0.05, 0.05]
        r.append(a / np.sum(a))
    for i in range(3):
        a = [0, 0.23, 0.17, 0.15, 0.05, 0.05, 0.1, 0.05]
        r.append(a / np.sum(a))
    for i in range(3):
        a = [0, 0.25, 0.1, 0.15, 0.05, 0.1, 0.15, 0.05]
        r.append(a / np.sum(a))
    for i in range(3):
        a = [0, 0.32, 0.12, 0.05, 0.05, 0.05, 0.05, 0.05]
        r.append(a / np.sum(a))
    return np.array(r)


def h_diff(a):
    return a + Q + D_ont


def h(a):
    h = np.zeros(12)
    diff = h_diff(a)
    h[0] = diff[0]
    for i in range(1, 12):
        h[i] = h[i - 1] + diff[i]
    return h


def U(hs):
    r = []
    r.append(U_vec(hs[0], 0) @ weights()[0])
    for j in range(1, 12):
        r.append(U_vec(hs[j], hs[j - 1]) @ weights()[j])
    R_sum = np.sum(np.array(r))
    R_sum -= np.exp(0.075 * (74.8 - np.mean(hs)))
    print(R_sum)
    return R_sum


def obj(argument):
    return -U(h(argument))


x0 = -1 * np.ones(12)
result = minimize(obj, x0, method='Nelder-Mead', bounds=[(-1.1, -0.9)] * 12, options={'maxiter': 10**5})


opt_diff = result.x
print(opt_diff)
LWL_ont = get_ont()
opt_h = h(opt_diff) + np.ones(12) * LWL_ont[0]
plt.plot(range(1, 13), LWL_ont, color=deep_cyan)
plt.plot(range(1, 13), opt_h, color=light_green)
plt.show()




