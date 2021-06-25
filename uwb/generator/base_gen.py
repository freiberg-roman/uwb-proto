import numpy as np


class BaseGenerator:
    """
    Base class for generators
    """

    def __init__(self):
        pass

    def gen(self):
        """
        Initializes the generation process.
        """
        return np.array([], dtype=np.float)

    def __iter__(self):
        """
        Returns the data per measurement location. General layout should be (np.array, (position)),
        where the position should match the shape of the underlying structure.
        """
        pass

    @property
    def shape(self):
        """
        Shape of the underlying structure. This will be used for pre-allocation.
        """
        return (1,)
