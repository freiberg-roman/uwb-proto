from functools import reduce
from itertools import product
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_blobs

from uwb.generator.base_gen import BaseGenerator


class BlobGenerator(BaseGenerator):
    """
    Simple nD-generator for an rectangular grid of measurments. For each
    position i*step_size x j*step_size for i,j in 1..grid_width // step_size,
    1..grid_length // step_size measurments_per_location many measurments are
    simulated by a k-blobs where k is randomly chosen from modal_range
    """

    def __init__(
        self,
        grid_dims: List[int],
        step_size: int,
        measurements_per_location: int,
        modal_range: Tuple[int, int],
        deviation: float = 10.0,
    ):
        super().__init__()
        self.step = step_size
        self.amount = measurements_per_location
        self.range = modal_range
        self.deviation = deviation
        self.grid_dims = grid_dims
        self.grid = []

        for dim in grid_dims:
            self.grid.append((np.arange(dim) + 1) * step_size)

    def gen(self) -> np.ndarray:
        prod = reduce((lambda x, y: x * y), self.grid_dims)  # multiplies all dimensions
        samples = np.zeros(self.grid_dims + [self.amount, len(self.grid_dims)])
        clusters = np.random.randint(
            self.range[0], self.range[1] + 1, size=self.grid_dims
        )

        mean = np.array(np.meshgrid(*self.grid, indexing="ij")).reshape(
            prod, len(self.grid_dims)
        )
        noise = (
            np.random.randn(self.range[1], prod, len(self.grid_dims)) * self.deviation
        )
        centers = (noise + mean).reshape(
            [self.range[1]] + self.grid_dims + [len(self.grid_dims)]
        )

        # transpose hack for selection
        roll_idx = np.roll(np.arange(centers.ndim), -1).tolist()
        centers = np.transpose(centers, roll_idx)

        for idxs in product(*[range(i) for i in self.grid_dims]):
            samples[idxs] = make_blobs(
                n_samples=self.amount, centers=(centers[idxs][:, 0 : clusters[idxs]]).T
            )[0]
        return samples

    def get_closest_position(self, coordinates):
        """Finds the closest positions in the grid map."""
        pass
