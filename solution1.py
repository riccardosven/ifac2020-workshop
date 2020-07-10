import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from scipy.optimize import minimize

from workshop.phase1 import load_data

# Load data
u, y = load_data()
N = len(u)
n = 200

# Create regression matrix
U = toeplitz(u, np.zeros(n))

# Linear regression
g_ls = np.linalg.lstsq(U, y, rcond=None)[0]

# Import true system
from workshop.phase1 import true_system as g_0

# Plot results
plt.plot(g_0, label="true")
plt.plot(g_ls, label="least-squares")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("impulse response")
plt.legend()
plt.show()

# Define stable-spline kernel
def stable_spline(theta):
    row = np.arange(n)[np.newaxis]
    return theta[0] * theta[1] ** np.maximum(row, row.T)


# Kernel-based estimation (posterior mean)
K = stable_spline([5, 0.9])
g_ker = K @ U.T @ np.linalg.solve(U @ K @ U.T + np.eye(N), y)

# Plot results
plt.plot(g_0, label="true")
plt.plot(g_ls, label="least-squares")
plt.plot(g_ker, label="kernel")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("impulse response")
plt.legend()
plt.show()


# Import negative marginal likelihood function
from workshop.phase1 import likelihood_function

neg_marg_lik = likelihood_function(U, y)

# Maximize marginal likelihood
sol = minimize(neg_marg_lik, [10.0, 0.9, 1.0], method="BFGS")
hypers = sol.x

# Kernel-based estimation
K_opt = stable_spline(hypers[:2])
g_opt = K_opt @ U.T @ np.linalg.solve(U @ K_opt @ U.T + hypers[2] * np.eye(N), y)


# Plot results
plt.plot(g_0, label="true")
plt.plot(g_ker, label="kernel")
plt.plot(g_opt, label="marg-lik")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("impulse response")
plt.legend()
plt.show()

# Print measures of fit
from workshop.utils import fit

print(" LS: {}".format(fit(g_ls, g_0)))
print("Ker: {}".format(fit(g_ker, g_0)))
print("Opt: {}".format(fit(g_opt, g_0)))
