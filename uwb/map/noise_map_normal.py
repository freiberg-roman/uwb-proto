import numpy as np

from uwb.generator import BaseGenerator
from uwb.map import NoiseMap


class NoiseMapNormal(NoiseMap):
    def __init__(self, measurement_generator: BaseGenerator):
        self.measurement_generator = measurement_generator
        self.means = np.zeros(measurement_generator.shape)
        self.covs = np.zeros(
            measurement_generator.shape + (measurement_generator.shape[-1],)
        )

    def gen(self):
        """Calculates estimates for a gaussian estimate"""
        for (samples, idxs, pos) in self.measurement_generator:
            self.means[idxs[:-1]] = samples.mean(axis=0) - pos
            self.covs[idxs[:-1]] = np.cov(samples.T)
