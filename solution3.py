import numpy as np
from matplotlib import pyplot as plt
from sklearn import gaussian_process as gp

from workshop.phase3 import load_data
from workshop.utils import split

x, y = load_data()

plt.plot(x, y)
plt.xlabel("Year")
plt.ylabel("CO2")
plt.title("Mauna Lua")
plt.show()

for ker in [
    gp.kernels.DotProduct() * gp.kernels.DotProduct() + gp.kernels.WhiteKernel(),
    gp.kernels.DotProduct() * gp.kernels.DotProduct()
    + gp.kernels.WhiteKernel()
    + gp.kernels.RationalQuadratic(),
    gp.kernels.DotProduct() * gp.kernels.DotProduct()
    + gp.kernels.WhiteKernel()
    + gp.kernels.RationalQuadratic()
    + gp.kernels.ExpSineSquared(length_scale=1.0),
]:

    mdl = gp.GaussianProcessRegressor(kernel=ker, normalize_y=True)
    mdl.fit(x, y)

    y_hat = mdl.predict(x)

    plt.plot(x, y)
    plt.plot(x, y_hat)
    plt.show()


ker = (
    gp.kernels.RBF(length_scale=50)
    + gp.kernels.RBF()
    * gp.kernels.ExpSineSquared(
        length_scale=1, periodicity=1, periodicity_bounds="fixed"
    )
    + gp.kernels.WhiteKernel()
)

mdl = gp.GaussianProcessRegressor(kernel=ker, normalize_y=True)
mdl.fit(x, y)

x_new = np.concatenate([x, np.linspace(x[-1], x[-1] + 3, 7)])
print(x_new.shape)

y_new, s_new = mdl.predict(x_new, return_std=True)
plt.plot(x, y)
plt.plot(x_new, y_new)
plt.fill_between(
    x_new.flatten(), y_new - 3 * s_new, y_new + 3 * s_new, color="gray", alpha=0.5
)
plt.show()
