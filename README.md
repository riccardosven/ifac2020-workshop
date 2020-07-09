# ifac2020-workshop
Practical session on kernel-based methods for learning dynamical systems

# Phase 0: Set up environment
Optional: create a virtual environment for the workshop
```
python -m venv .venv --prompt ifac2020
source .venv/bin/activate
```

Install the packages required
```
pip install -r requirements.txt
```

In all cases, we use
```
import numpy as np
```
at the beginning of each python file or session.

# Phase 1: Kernel-based identification in numpy
In this phase, we will solve the system identification problem from scratch using `numpy`:

### Loading the data
The data can be loaded from the `workshop` package using the `load_system_data`
function:
```
from workshop import load_system_data

u, y = load_data()
N = len(u)
```
The `u` variable now contains the input signal, the `y` variable contains the
output signal. (The system itself is a high-order linear dynamical system.)

### Try linear regression
Create the toeplitz matrix of the input
```
from scipy.linalg import toeplitz
n = 200
U = toeplitz(u, np.zeros(n))
```

Estimate the impulse response using a pseudoinverse (linear regression)
```
g_ls = np.linalg.lstsq(U, y, rcond=None)[0]
```

Now get the true impulse response of the system
```
from workshop.phase1 import true_system as g_0
```

Plot the results and compare with the true impulse response
```
import matplotlib.pyplot as plt
plt.plot(g_0, label="true")
plt.plot(g_ls, label="least-squares")
plt.title("Linear system")
plt.xlabel("lag")
plt.ylabel("Impulse response")
plt.show()
```

### Define the stable-spline kernel
We will use the first-order stable-spline kernel
```

def stable_spline(theta):
    row = np.arange(n)[np.newaxis]
    return theta[0] * theta[1] ** np.maximum(row, row.T)
```
where `n` is the length of the impulse response and `theta = (theta1, theta2)`
are the hyperparameters.

We can now implement the kernel-based estimation with
```
K = stable_spline([10, 0.9])
g_ker = K@U.T@np.linalg.solve(U@K@U.T + np.eye(N), y)
```
Were we have set the hyperparameters arbitrarily.

Plot the results and compare with the least-squares estimate.
Note how regularization has reduced the variance (especially in the tail of the
impulse response).


### Estimate the hyperparameters
We will now use marginal likelihood maximization to find the optimal
hyperparameters for this problem.

Define the marginal likelihood function 

```
from workshop.phase1 import likelihood_function
neg_marg_lik = likelihood_function(U, y)
```

and optimize the criterion starting from the previous guess:
```
sol = minimize(neg_marg_lik, [10.0, 0.9, 1.0])
hypers = sol.x


K_opt = stable_spline(hypers[:2])
g_opt = K_opt@U.T@np.linalg.solve(U@K_opt@U.T + hypers[2]*np.eye(N), y)
```

*WARNING: this optimization may take a while!*

Plot the results and compare with the previous estimates. Note that the fit is
a bit higher in this optimized case (even though there is a higher variance in
the estimate):

```
from workshop.utils import fit
print(" LS: " + str(fit(g_ls, g_0)))
print("Ker: " + str(fit(g_ker, g_0)))
print("Opt: " + str(fit(g_opt, g_0)))
```

# Phase 2: Gaussian process models in sklearn
We will now use the GP module of the sklearn package.

```
import sklearn.gaussian_processes as gp
import numpy as np
np.random.seed(1337)
```

Import the true function for this phase and create some observations from the
function (that will act as our data)
```
from workshop.phase2 import true_function

N = 16 # Number of data
x_obs = 2*np.random.rand(N) # Function defined on [0,2]
y_obs = true_function(x_obs)
```

### Draw realizations from the prior
Now, we define some kernels that we will experiment with
```
K_linear = gp.kernels.DotProduct()
K_quadratic = K_linear*K_linear
K_gauss = gp.kernels.RBF()
K_laplace = gp.kernel.Matern()
```
The interface to the GP using these kernels is accessed using the
`GaussianProcessRegressor` class:
```
mdl = gp.GaussianProcessRegressor(kernel=kernel)
```

We can draw some realizations from these kernels (effectively: the priors) to
see the possible functions they represent
```
for i, ker in enumerate([K_linear, K_quadratic, K_gauss, K_laplace]):
	ax = plt.subplot("22"+str(i+1))
	mdl = gp.GaussianProcessRegressor(kernel=ker)
	for n in range(7):
		ax.plot(x, mdl.predict(x.reshape(-1,1), random_state=n))
		ax.set_title(["Linear", "Quadratic", "Gaussian", "Laplace"][i])
plt.show()
```

### Fit GP models to available data
We can now use these kernels to fit a regression model to the available data:
```
mdl = gp.GaussianProcessRegressor(kernel=ker)
mdl.fit(x_obs.reshape(-1,1), y_obs)
post_mean = mdl.predict(x.reshape(-1,1))

plt.plot(x, y, linestyle='--', label="True")
plt.scatter(x_obs, y_obs, color='r', label="Observations")
plt.plot(x, post_mean, label="Posterior mean")
plt.show()
```

In addition to the posterior mean (returned by the `predict` method) we can
plot the credible intervals by returing the standard deviation of the
prediction
```
post_mean, post_std = mdl.predict(x.reshape(-1,1), return_std=True)
plt.plot(x, y, linestyle='--', label="True")
plt.scatter(x_obs, y_obs, color='r', label="Observations")
plt.plot(x, post_mean, label="Posterior mean")
plt.fill_between(x, post_mean - 3*post_std, post_mean+ 3*post_std, color="gray", alpha=0.5)
plt.show()
```

### Introduce noise
The previous model was an interpolation model (no noise). Introduce some noise
in the observations:
```
x_meas = x_obs
y_meas = y_obs + 0.1*np.random.randn(N)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_meas, y_meas, color="r", label="Measurements")
plt.legend()
plt.show()
```

Note that an interpolating solution will not fit the measurements!
```
mdl = gp.GaussianProcessRegressor(kernel=K_gauss)
mdl.fit(x_meas.reshape(-1, 1), y_meas)
post_mean, post_std = mdl.predict(x.reshape(-1,1), return_std=True)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_meas, y_meas, color="r", label="Measurements")
plt.plot(x, post_mean, label="Posterior mean")
plt.fill_between(x, post_mean - 3*post_std, post_mean+ 3*post_std, color="gray", alpha=0.5)
plt.legend()
plt.show()
```

We can however use a `WhiteKernel` to represent iid Gaussian measurement noise
```
K_smoothing = K_gauss + gp.kernels.WhiteKernel()
mdl = gp.GaussianProcessRegressor(kernel = K_smoothing)
mdl.fit(x_meas.reshape(-1, 1), y_meas)
post_mean, post_std = mdl.predict(x.reshape(-1,1), return_std=True)
plt.plot(x, y, linestyle="--", label="True")
plt.scatter(x_meas, y_meas, color="r", label="Measurements")
plt.plot(x, post_mean, label="Posterior mean")
plt.fill_between(x, post_mean - 3*post_std, post_mean+ 3*post_std, color="gray", alpha=0.5)
plt.legend()
plt.show()
```

