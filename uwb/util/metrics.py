import numpy as np


def ess(weights):
    """Computes effective sample size."""

    M = len(weights)
    CV = cv(weights)
    return M / (1 + CV)


def cv(weights):
    """Computes coefficient of variation."""

    M = len(weights)
    return np.mean((M * weights - 1) ** 2)
