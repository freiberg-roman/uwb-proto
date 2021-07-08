import numpy as np
from scipy.stats import multivariate_normal

from uwb.map import NoiseMap


class NoiseMapNormal(NoiseMap):
    """Generates noise map with uni-modal Gaussian estimates.

    For given generator (see :class:`uwb.generator.BaseGenerator`) of arbitrary dimension,
    this class provides a noise map with sampling and conditional probabilities functionality.

    Attributes:
        generator: generator that provides measurements for given position
          this class must be iterable with valid format of (samples, idx, position).
    """

    def __init__(self, generator):
        """Inits and allocates numpy arrays for parameters."""
        super().__init__(generator)
        self.means = np.zeros(generator.shape + (len(generator.shape),))
        self.covs = np.zeros(
            generator.shape + (len(generator.shape),) + (len(generator.shape),)
        )

    def gen(self):
        """Calculates estimates for a gaussian distribution"""
        for (samples, idxs, pos) in self.generator:
            self.means[idxs] = samples.mean(axis=0) - pos
            self.covs[idxs] = np.cov(samples.T)

    def conditioned_probability(self, z, particles):
        """Computes conditioned probabilities.

        Computes the conditioned probabilities p(z|x) where x is given by the nearest map position
        in the map for the provided samples.

        Args:
            z: Numpy array of measurements with format (N,d).
            particles: particles from the particle filter used for density estimation.
        """
        pos_coord, pos = self.generator.get_closest_position(particles)
        probs = np.empty(len(pos))

        # couldn't find a faster way.. suggestions are appreciated
        for i, (measurement, p, mean, cov) in enumerate(
            zip(
                z,
                pos,
                self.means[tuple(pos_coord.T.tolist())],
                self.covs[tuple(pos_coord.T.tolist())],
            )
        ):
            probs[i] = multivariate_normal.pdf(measurement, mean=mean + p, cov=cov)

        return probs

    def sample_from(self, coordinates):
        """Samples particles from a normal distribution.

        Samples particles for given coordinates from a normal distribution.

        Args:
            coordinates: particles to find nearest positions from, which are used for sampling.
        """
        samples = np.empty((len(coordinates), self.means.shape[-1]), dtype=float)
        pos_coord, pos = self.generator.get_closest_position(coordinates)
        for i, (p, mean, cov) in enumerate(
            zip(
                pos,
                self.means[tuple(pos_coord.T.tolist())],
                self.covs[tuple(pos_coord.T.tolist())],
            )
        ):
            samples[i] = multivariate_normal.rvs(mean=mean, cov=cov) + p

        return samples
