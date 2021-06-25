from uwb.generator import BaseGenerator
from uwb.map import NoiseMap


class NoiseMapGM(NoiseMap):
    """Provides estimates for each position with Gaussian Mixtures via DBSCAN preprocessing."""

    def __init__(self, measurement_generator: BaseGenerator):
        """Calculates estimates.

        If DBSCAN finds a cluster with less than three samples, they will be thrown away.
        """
        self.measurement_generator = measurement_generator
