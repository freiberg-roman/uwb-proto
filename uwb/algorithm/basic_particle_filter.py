import numpy as np
from scipy.stats import multivariate_normal

from uwb.algorithm import ParticleFilter


class BasicParticleFilter(ParticleFilter):
    """Implementation of Basic Particle Filter

    Provides particle weight update and resampling as described in paper
    `W. Suski <https://ieeexplore.ieee.org/document/6514113>`

    Attributes:
        init_particles: Numpy array of initial particle distribution with format (N, d) where N, d
          is the number of particles and d the dimension respectively.
        init_weights: Numpy array of normalized weights with format (N,).
    """

    def __init__(self, init_particles, init_weights):
        """Initialized and computes data covariance."""
        super().__init__(init_particles, init_weights)
        self.data_cov = np.cov(init_particles)

    def update_weight(self, z):
        """Update weights particles as means and covariance of data distribution

        Args:
            z: measurements collected by sensor for weight updates
                expected shape (N, d) where N is the batch size
        """
        M = len(self.particles)
        z = np.tile(z, (M, 1, 1))
        cov = np.tile(self.data_cov, (M, 1))
        self.weights = self.weights * np.prod(
            multivariate_normal.pdf(z, mean=self.samples, cov=cov), axis=1
        )  # iid assumption and product over batch dimension

        # normalize weights
        self.weights = self.weights / (np.sum(self.weights) + 1e-10)

    def resample(self):
        """Resampling according to weights of particles"""
        M = len(self.samples)
        acc_weights = np.cumsum(self.weights)
        acc_weights[-1] = 1  # in case of rounding errors

        uniform_samples = np.random.uniform(M)  # samples uniform from [0,1)
        positions = np.searchsorted(
            acc_weights, uniform_samples
        )  # find matching positions
        self.particles = multivariate_normal.rvs(
            mean=self.samples[positions], cov=self.data_cov
        )

        # update covariance
        self.data_cov = np.cov(self.samples)
        self.weights = np.ones_like(self.weights) * (1 / M)
