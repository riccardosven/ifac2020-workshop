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
at the beginning of each python file or session

# Phase 1: Kernel-based identification in numpy
In this phase, we will solve the system identification problem from scratch using `numpy`:

### Loading the data
The data can be loaded from the `workshop` package using the `load_system_data`
function:
```
from workshop import load_system_data

data = load_system_data()
u = data["u"]
y = data["y"]
g_0 = data["g_0"]

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
g_hat = np.linalg.lstsq(U, y, rcond=None)[0]
```

Plot the results and compare with the true impulse response
```
import matplotlib.pyplot as plt
plt.plot(g_0, label="true")
plt.plot(g_hat, label="lin-reg")
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

*WARNING:* this optimization may take a while!

Plot the results and compare with the previous estimates. Note that the fit is
a bit higher in this optimized case (even though there is a higher variance in
the estimate):

```
from workshop.phase1 import fit
print(" LS: " + str(fit(g_ls, g_0)))
print("Ker: " + str(fit(g_ker, g_0)))
print("Opt: " + str(fit(g_opt, g_0)))
```
