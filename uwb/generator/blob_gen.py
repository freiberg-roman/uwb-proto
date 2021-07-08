from functools import reduce
from itertools import product
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_blobs

from uwb.generator.base_gen import BaseGenerator


class BlobGenerator(BaseGenerator):
    """N-Dimensional Gaussian Mixture generator for a grid map.

    Generator provides for predefined N dimensions a grid in a N dimensional cube. For each
    position in the cube a fixed amount of samples are generated using Gaussian Mixtures.
    The amount of mixture components can be specified in the :attr:`modal_range` of this class.

    Attributes:
        grid_dims: list of widths for N dimensions.
        step_size: one dimensional distance between measurement positions.
        measurements_per_location: number of measurements per position in grid.
        model_range: tuple with minimum and maximum number of clusters.
        deviation: standard deviation for Gaussian distribution.
    """

    def __init__(
        self,
        grid_dims: List[int],
        step_size: int,
        measurements_per_location: int,
        modal_range: Tuple[int, int],
        deviation: float = 10.0,
    ):
        """Pre-allocates data structures for generation."""
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

    def gen(self):
        """Initializes generation process."""
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
            [self.range[1]] + list(self.grid_dims) + [len(list(self.grid_dims))]
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
        """Finds the closest positions in the grid map.

        Finds the closest (L2-norm) position in the grid.

        Args:
            coordinates: Numpy array (N, d) where N, d are batch size and dimensions respectively.
        """
        assert coordinates.shape[1] == len(self.grid_dims)
        pos = np.empty((coordinates.shape[0], len(self.grid_dims)))

        for i in range(len(self.grid_dims)):
            i_pos = np.searchsorted(self.grid[i], coordinates[:, i], side="left")
            i_valid = (i_pos != 0) & (i_pos < len(self.grid[0]))
            i_pos = np.clip(i_pos, 0, len(self.grid[i]) - 1)
            i_dist_right = self.grid[i][i_pos] - coordinates[:, i]
            i_dist_left = coordinates[:, i] - self.grid[i][i_pos - 1]
            i_pos[i_valid & (i_dist_right > i_dist_left)] -= 1

            pos[:, i] = i_pos

        return pos.astype(int), (pos + 1) * self.step

    def __iter__(self):
        """Provides iterator for samples. Generation will be performed if not invoked previously"""
        if self._data is None:
            self.gen()
        return self

    def __next__(self):
        """Measurements for next position in grid."""
        if self._iter is None:
            self._iter = product(*[range(i) for i in self.grid_dims])
        try:
            idx = next(self._iter)
        except StopIteration:
            self._iter = None
            raise StopIteration
        return (
            self._data[idx],
            idx,
            (np.array(idx) + 1) * self.step,
        )  # samples, index, position

    @property
    def shape(self):
        """Underlying shape of data."""
        return tuple(self.grid_dims)
