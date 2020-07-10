import numpy as np
from matplotlib import pyplot as plt
from sklearn import gaussian_process as gp

from workshop.phase2 import true_function

# Set the seed
np.random.seed(1337)

# Get some observations
N = 16
x_obs = 2 * np.random.rand(N)
y_obs = true_function(x_obs)

# Plot the true function
x = np.linspace(0, 2, 100)
y = true_function(x)

plt.plot(x, y, linestyle="--")
plt.scatter(x_obs, y_obs, color="r")
plt.title("True function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# Define kernels
K_linear = gp.kernels.DotProduct()
K_quadratic = gp.kernels.Product(K_linear, K_linear)
K_gauss = gp.kernels.RBF()
K_laplace = gp.kernels.Matern()

# Plot some realizations from the kernels
for i, ker in enumerate([K_linear, K_quadratic, K_gauss, K_laplace]):
    ax = plt.subplot("22" + str(i + 1))
    mdl = gp.GaussianProcessRegressor(kernel=ker)
    for n in range(7):
        ax.plot(x, mdl.sample_y(x.reshape(-1, 1), random_state=n))
        ax.set_title(["Linear", "Quadratic", "Gaussian", "Laplace"][i])
plt.show()

# Fit the GP regression model to the data
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_obs, y_obs, color="r", label="Observations")
for i, ker in enumerate([K_linear, K_quadratic, K_gauss, K_laplace]):
    mdl = gp.GaussianProcessRegressor(kernel=ker)
    mdl.fit(x_obs.reshape(-1, 1), y_obs)
    plt.plot(
        x,
        mdl.predict(x.reshape(-1, 1)),
        label=["Linear", "Quadratic", "Gaussian", "Laplace"][i],
    )
plt.legend()
plt.show()

print("Kernel: {}".format(mdl.kernel_))
print("Parameters: {}".format(mdl.kernel_.theta))
print("Log-likelihood: {}".format(mdl.log_marginal_likelihood(mdl.kernel_.theta)))


# Show credible interval
mdl = gp.GaussianProcessRegressor(kernel=K_laplace)
mdl.fit(x_obs.reshape(-1, 1), y_obs)
post_mean, post_stddev = mdl.predict(x.reshape(-1, 1), return_std=True)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_obs, y_obs, color="r", label="Observations")
plt.plot(x, post_mean, label="Posterior mean")
plt.fill_between(
    x,
    post_mean - 3 * post_stddev,
    post_mean + 3 * post_stddev,
    color="gray",
    alpha=0.5,
    label="99% CI",
)
plt.legend()
plt.show()

# Introduce Noise
x_meas = x_obs
y_meas = y_obs + 0.1 * np.random.randn(N)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_meas, y_meas, color="r", label="Measurements")
plt.legend()
plt.show()

# Overfitting posterior mean
mdl = gp.GaussianProcessRegressor(kernel=K_gauss)
mdl.fit(x_meas.reshape(-1, 1), y_meas)
post_mean, post_std = mdl.predict(x.reshape(-1, 1), return_std=True)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_meas, y_meas, color="r", label="Measurements")
plt.plot(x, post_mean, label="Posterior mean")
plt.fill_between(
    x,
    post_mean - 3 * post_std,
    post_mean + 3 * post_std,
    color="gray",
    alpha=0.5,
    label="99% CI",
)

plt.legend()
plt.show()

# Using a noise model
K_smoothing = K_gauss + gp.kernels.WhiteKernel()
mdl = gp.GaussianProcessRegressor(kernel=K_smoothing)
mdl.fit(x_meas.reshape(-1, 1), y_meas)
post_mean, post_std = mdl.predict(x.reshape(-1, 1), return_std=True)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_meas, y_meas, color="r", label="Measurements")
plt.plot(x, post_mean, label="Posterior mean")
plt.fill_between(
    x,
    post_mean - 3 * post_std,
    post_mean + 3 * post_std,
    color="gray",
    alpha=0.5,
    label="99% CI",
)
plt.legend()
plt.show()
