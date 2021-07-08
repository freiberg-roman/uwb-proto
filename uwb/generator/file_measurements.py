import numpy as np
from numpy import genfromtxt


class FileMeasurements:
    """Reads in measurement batches from specified file.

    Arguments:
        file_name: name of file
        batch_size: batch size of measurements
    """

    def __init__(self, file_name, batch_size):
        measurements = genfromtxt(file_name, delimiter=",")
        self.idx = 0
        self.batch_size = batch_size
        self.measurements = np.array_split(0, len(measurements), batch_size)

    def __next__(self):
        if self.idx >= len(self.measurements):
            raise StopIteration

        return self.measurements[self.idx]
