import numpy as np

from uwb.algorithm.particle_filter import ParticleFilter
from uwb.map import NoiseMap


class MNMAParticleFilter(ParticleFilter):
    """Implementation of Measurement Noise Map Augmented Particle Filter.

    Core component of Measurement Noise Map Augmented Particle Filter pipeline. It provides logic
    for weight updates and resampling as described in paper
    `W.Sluski <https://ieeexplore.ieee.org/document/6514113>`

    Attributes:
        init_particles: initial positions for particles
        init_weights: initial weights for particles
        map: Noise Map for location which was previously empirically estimated.
    """

    def __init__(self, init_particles, init_weights, map: NoiseMap):
        """Initializes particles, weights and noise map."""
        super().__init__(init_particles, init_weights)
        self.map = map

    def update_weights(self, z):
        """Updates weights according to map noise estimations"""
        for i in range(len(self.particles)):
            self.weights[i] *= np.prod(
                self.map.conditioned_probability(
                    z, np.tile(self.particles[i], (len(z), 1))
                )
            )  # iid assumption

        # normalize weights ... numerical instabilities may happen here
        self.weights = self.weights / np.sum(self.weights)

    def resample(self):
        """Resamples particles."""
        M = len(self.particles)
        acc_weights = np.cumsum(self.weights)
        acc_weights[-1] = 1  # in case of rounding errors

        uniform_samples = np.random.uniform(0.0, 1.0, M)  # samples uniform from [0,1)
        positions = np.searchsorted(
            acc_weights, uniform_samples
        )  # find matching positions
        self.particles = self.map.sample_from(self.particles[positions])
        self.weights = np.ones_like(self.weights) * (1 / M)
