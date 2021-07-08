import numpy as np


class RngSensorMeasurements:
    """Uniform measurements in a grid space.

    This class is just for testing purposes. Similar interface structure is expected for
    sensor input.
    """

    def __init__(self, ranges, amount, dim):
        """Initializes ranges."""
        self.ranges = ranges
        self.amount = amount
        self.dim = dim

    def __next__(self):
        return np.tile(
            np.random.uniform(
                low=self.ranges[0],
                high=self.ranges[1],
                size=(self.amount, self.dim),
            ),
            (1, 1),
        )
