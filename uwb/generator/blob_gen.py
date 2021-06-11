from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs

from uwb.generator.base_gen import BaseGenerator


class BlobGenerator(BaseGenerator):
    """
    Simple 2D-generator for an rectangular grid of measurments. For each
    position i*step_size x j*step_size for i,j in 1..grid_width // step_size,
    1..grid_length // step_size measurments_per_location many measurments are
    simulated by a k-blobs where k is randomly chosen from modal_range
    """

    def __init__(
        self,
        grid_length: int,
        grid_width: int,
        step_size: int,
        measurements_per_location: int,
        modal_range: Tuple[int, int],
        deviation: float = 10.0,
    ):
        super().__init__()
        self.length = grid_length
        self.width = grid_width
        self.step = step_size
        self.amount = measurements_per_location
        self.range = modal_range
        self.deviation = deviation

    def gen(self) -> np.ndarray:
        width = self.width // self.step
        length = self.length // self.step
        samples = np.zeros((width, length, self.amount, 2))

        clusters = np.random.randint(
            self.range[0], self.range[1] + 1, size=(width, length)
        )

        # calculate centers
        grid_width = (np.arange(width) + 1) * self.step
        grid_length = (np.arange(length) + 1) * self.step
        mean = np.array(
            [
                np.repeat(grid_width, len(grid_length)),
                np.tile(grid_length, len(grid_width)),
            ]
        ).T
        noise = np.random.randn(self.range[1], width * length, 2) * self.deviation
        centers = (noise + mean).reshape((self.range[1], width, length, 2))

        for i in range(width):
            for j in range(length):
                samples[i, j, :] = make_blobs(
                    n_samples=self.amount, centers=centers[0 : clusters[i, j], i, j, :]
                )[0]

        return samples
