import numpy as np


def ess(weights):
    """
    note: array of shape (M,) of weights
    """

    M = len(weights)
    CV = cv(weights)
    return M / (1 + CV)


def cv(weights):
    M = len(weights)
    return np.mean((M * weights - 1) ** 2)
