import numpy as np


class BaseGenerator:
    """
    Base class for generators
    """

    def __init__(self):
        pass

    def gen(self) -> np.ndarray:
        return np.array([], dtype=np.float)
