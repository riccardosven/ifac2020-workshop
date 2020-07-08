from workshop.phase1 import load_data
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

data = load_data()
u = data["u"]
y = data["y"]
g_0 = data["g_0"]

N = len(u)
n = 200
U = toeplitz(u, np.zeros(n))

g_ls = np.linalg.lstsq(U, y, rcond=None)[0]

plt.plot(g_0, label="true")
plt.plot(g_ls, label="lin-reg")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("impulse response")
plt.show()


def stable_spline(theta):
    row = np.arange(n)[np.newaxis]
    return theta[0] * theta[1] ** np.maximum(row, row.T)

K = stable_spline([10, 0.9])
g_ker = K@U.T@np.linalg.solve(U@K@U.T + np.eye(N), y)

plt.plot(g_0, label="true")
plt.plot(g_ls, label="lin-reg")
plt.plot(g_ker, label="kernel")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("impulse response")
plt.show()


from workshop.phase1 import likelihood_function
neg_marg_lik = likelihood_function(U, y)

sol = minimize(neg_marg_lik, [10.0, 0.9, 1.0])
hypers = sol.x

K_opt = stable_spline(hypers[:2])
g_opt = K_opt@U.T@np.linalg.solve(U@K_opt@U.T + hypers[2]*np.eye(N), y)


plt.plot(g_0, label="true")
plt.plot(g_ker, label="kernel")
plt.plot(g_opt, label="marg-lik")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("impulse response")
plt.show()

from workshop.phase1 import fit
print(" LS: " + str(fit(g_ls, g_0)))
print("Ker: " + str(fit(g_ker, g_0)))
print("Opt: " + str(fit(g_opt, g_0)))
