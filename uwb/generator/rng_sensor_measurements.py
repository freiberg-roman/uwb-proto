import numpy as np


class RngSensorMeasurements:
    """Uniform measurements in a grid space.

    This class is just for testing purposes. Similar interface structure is expected for
    sensor input.
    """

    def __init__(self, ranges, amount):
        """Initializes ranges."""
        self.ranges = ranges
        self.amount = amount

    def __next__(self):
        return np.random.uniform(
            low=self.ranges[0],
            high=self.ranges[1],
            size=(self.amount, len(self.ranges)),
        )
