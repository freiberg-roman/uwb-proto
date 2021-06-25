from copy import deepcopy

from uwb.generator import BaseGenerator
from uwb.map import NoiseMap


class NoiseMapGM(NoiseMap):
    """Provides estimates for each position with Gaussian Mixtures via DBSCAN preprocessing."""

    def __init__(self, measurement_generator: BaseGenerator):
        """Pre-allocates data structures for parameter estimations.

        This module is just a prove of concept and has to be rewritten if the task requires speed.
        The simpler TODO ref class noise map normal should be used for now.
        """
        self.measurement_generator = measurement_generator

        def _rec_array(shapes):
            if not shapes:
                return []
            else:
                first = shapes.pop(0)
                return [_rec_array(deepcopy(shapes)) for _ in range(first)]

        self.params = _rec_array(list(measurement_generator.shape))
        print(self.params)

    def gen(self):
        """Calculates estimates.

        If DBSCAN finds a cluster with less than three samples, they will be thrown away.
        """
        pass
