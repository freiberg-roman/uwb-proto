import numpy as np
from scipy.stats import multivariate_normal

from uwb.algorithm.particle_filter import ParticleFilter


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
        self.data_cov = np.cov(init_particles.T)

    def update_weight(self, z):
        """Update weights particles as means and covariance of data distribution

        Args:
            z: measurements collected by sensor for weight updates
                expected shape (N, d) where N is the batch size
        """
        # unfortunately normal implementation doesn't seem to support batch dim
        # needs to be reimplemented for performance reasons
        for item in z:
            for i, p in enumerate(self.particles):
                self.weights[i] *= multivariate_normal.pdf(
                    item, mean=p, cov=self.data_cov
                )
            self.weights = self.weights / np.sum(self.weights)

        # normalize weights

    def resample(self):
        """Resampling according to weights of particles"""
        M = len(self.particles)
        acc_weights = np.cumsum(self.weights)
        acc_weights[-1] = 1  # in case of rounding errors

        uniform_samples = np.random.uniform(0.0, 1.0, M)  # samples uniform from [0,1)
        positions = np.searchsorted(
            acc_weights, uniform_samples
        )  # find matching positions

        # again same problem as in update_weigth
        for i, p in enumerate(positions):
            self.particles[i] = multivariate_normal.rvs(
                mean=self.particles[p], cov=self.data_cov
            )

        # update covariance
        self.data_cov = np.cov(self.particles.T)
        self.weights = np.ones_like(self.weights) * (1 / M)
