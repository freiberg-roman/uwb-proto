import os

import numpy as np
from numpy import genfromtxt


class FileMeasurements:
    """Reads in measurement batches from specified file.

    Arguments:
        file_name: name of file
        batch_size: batch size of measurements
    """

    def __init__(self, file_name, batch_size):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_name = ("%s/../../" % dir_path) + file_name

        measurements = genfromtxt(file_name, delimiter=",")
        self.idx = 0
        self.batch_size = batch_size
        self.measurements = np.array_split(measurements, batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.measurements):
            raise StopIteration

        ret = self.measurements[self.idx]
        self.idx += 1
        return ret
