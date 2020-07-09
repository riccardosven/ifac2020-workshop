import numpy as np


def fit(g_hat, g):
    "Normalized measure of fit"
    e = g - g_hat
    return 1 - np.linalg.norm(e) / np.linalg.norm(g)
