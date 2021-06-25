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
        self._data = None
        self._iter = None

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

        self._data = samples
        return samples

    def get_closest_position(self, coordinates):
        """Finds the closest positions in the grid map."""
        assert coordinates.shape[1] == len(self.grid_dims)
        pos = np.empty(coordinates.shape[0], len(self.grid_dims))

        for i in range(len(self.grid_dims)):
            i_pos = np.searchsorted(self.grid[i], coordinates[:, i], side="right")
            i_valid = (i_pos != 0) & (i_pos < len(self.grid[0]))
            i_pos = np.clip(i_pos, 0, len(self.grid[i]) - 1)
            i_dist_right = self.grid[i][i_pos] - coordinates[i]
            i_dist_left = coordinates[i] - self.grid[i][i_pos - 1]
            i_pos[i_valid & (i_dist_right > i_dist_left)] -= 1

            pos[:, i] = i_pos

        return pos

    def __iter__(self):
        if self._data is None:
            self.gen()
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = product(*[range(i) for i in self.grid_dims])
        idx = next(self._iter)
        return self._data[idx], idx

    @property
    def shape(self):
        return tuple(self.grid_dims)
