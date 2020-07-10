import numpy as np


def fit(g_hat, g):
    "Normalized measure of fit"
    e = g - g_hat
    return 1 - np.linalg.norm(e) / np.linalg.norm(g)


def split(x, fraction=0.5):
    n = int(fraction * len(x))
    return x[:n], x[n:]
