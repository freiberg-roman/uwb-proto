import numpy as np

from uwb.generator import BaseGenerator


class NoiseMap:
    """Base class for noise maps

    Attributes:
        generator: measurement that can be accessed by an iterator.
    """

    def __init__(self, generator: BaseGenerator):
        """Initializes the estimation of the parameters."""
        self.gen = generator

    def get_params(self, coordinates: np.array):
        """Returns list of lists with parameter estimates for the given coordinates.

        The return value is dependent on the underlying model for the noise map.
        """
        pass

    def conditioned_probability(self, z, particles):
        """Computes conditioned probabilities p(z|x)

        Computes conditioned probabilities where x is given from the map for each particle.
        """
        pass

    def sample_from(self, coordinates):
        """Samples from distributions of the map for given coordinates."""
        pass
