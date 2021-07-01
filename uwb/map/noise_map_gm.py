from copy import deepcopy
from functools import reduce

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN

from uwb.generator import BaseGenerator
from uwb.map import NoiseMap


class NoiseMapGM(NoiseMap):
    """Provides estimates for each position with Gaussian Mixtures via DBSCAN preprocessing."""

    def __init__(self, measurement_generator: BaseGenerator, eps=2, min_samples=3):
        """Pre-allocates data structures for parameter estimations.

        This module is just a prove of concept and has to be rewritten if the task requires speed.
        The simpler TODO ref class noise map normal should be used for now.
        """
        self.measurement_generator = measurement_generator
        self.db = DBSCAN(eps=eps, min_samples=min_samples)

        def _rec_array(shapes):
            if not shapes:
                return []
            else:
                first = shapes.pop(0)
                return [_rec_array(deepcopy(shapes)) for _ in range(first)]

        self.params = _rec_array(list(measurement_generator.shape))
        self._dim = sum(map(lambda x: 1, list(measurement_generator.shape)))

    def gen(self):
        """Calculates estimates.

        If DBSCAN finds a cluster with less than three samples, they will be thrown away.
        """
        for (samples, idxs, pos) in self.measurement_generator:
            db = self.db.fit(samples)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            param_list = reduce(lambda params, idx: params[idx], idxs, self.params)
            used_data = samples.shape[0] - n_noise

            weights = []
            means = []
            covariances = []

            for i in range(n_clusters):
                mask = labels == i
                means.append(samples[mask, :].mean(axis=0) - pos)
                covariances.append(np.cov(samples[mask, :].T))
                weights.append(mask.sum() / used_data)

            weights = np.stack(weights)
            means = np.stack(means)
            covariances = np.stack(covariances)
            param_list.append((weights, means, covariances))

    def sample_from(self, coordinates):
        positions = self.measurement_generator.get_closest_position(coordinates)
        samples = np.empty_like(positions)
        for i, p in enumerate(positions):
            weights, means, covs = self[p]
            selection = np.random.choice(np.arange(len(weights)), p=weights)
            samples[(i,) + p] = multivariate_normal.rvs(
                mean=means[selection], cov=covs[selection]
            )
        return samples

    def conditioned_probability(self, z, samples):
        positions = self.measurement_generator.get_closest_position(samples)
        prob = np.zeros((len(z)))
        for i, p in enumerate(positions):
            weights, means, covs = self[p]
            for w in enumerate(weights):
                prob += w * multivariate_normal.pdf(mean=means[i], cov=covs[i])
        return prob

    def __getitem__(self, item):
        if len(item) == self._dim:
            param_list = reduce(lambda params, idx: params[idx], item, self.params)
            return param_list[0]
