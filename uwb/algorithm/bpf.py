import numpy as np

from uwb.map import NoiseMapNormal


class BPF:
    def __init__(
        self, map: NoiseMapNormal, dyn_model, init_samples, init_vel, init_weights
    ):
        self.map = map
        self.dyn_model = dyn_model
        self.samples = init_samples
        self.vel = init_vel
        self.weights = init_weights

    def reweight(self, z):
        self.weights = self.weights * self.map.conditioned_probability(z, self.samples)

        # normalize weights
        self.weights = self.weights / (np.sum(self.weights) + 1e-10)

    def resample(self):
        M = len(self.samples)
        acc_weights = np.cumsum(self.weights)
        acc_weights[-1] = 1  # in case of rounding errors

        uniform_samples = np.random.uniform(M)  # samples uniform from [0,1)
        positions = np.searchsorted(
            acc_weights, uniform_samples
        )  # find matching positions
        self.samples = self.map.sample_from(self.samples[positions])
        self.weights = np.ones_like(self.weights) * (1 / M)
