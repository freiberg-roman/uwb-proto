from copy import deepcopy
from functools import reduce

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN

from uwb.generator import BaseGenerator
from uwb.map import NoiseMap


class NoiseMapGM(NoiseMap):
    """Provides estimates for each position with Gaussian Mixtures via DBSCAN preprocessing.

    This module is just a prove of concept and has to be rewritten for performance purposes.
    The simpler :class:`uwb.map.NoiseMapNormal` should be used in most cases.

    Attributes:
        generator: provides iterator for measurements
        eps: Optional; distance between samples for DBSCAN clustering.
        min_samples: Optional; min number of samples for a group to be considered a cluster.
    """

    def __init__(self, generator: BaseGenerator, eps=2, min_samples=3):
        """Pre-allocates data structures for parameter estimations."""
        super().__init__(generator)
        self.db = DBSCAN(eps=eps, min_samples=min_samples)

        def _rec_array(shapes):
            if not shapes:
                return []
            else:
                first = shapes.pop(0)
                return [_rec_array(deepcopy(shapes)) for _ in range(first)]

        self.params = _rec_array(list(generator.shape))
        self._dim = sum(map(lambda x: 1, list(generator.shape)))

    def gen(self):
        """Calculates estimates. (See paper W.Suski <https://ieeexplore.ieee.org/document/6514113>`)

        If DBSCAN finds a cluster with less than :attr:`min_samples` samples, they will be thrown
        away.
        """
        for (samples, idxs, pos) in self.generator:
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
        """Samples for each coordinate using a Gaussian Mixture.

        This is generally to slow for most applications. TODO C++/Numpy reimplementation.
        """
        pos_coords, pos = self.generator.get_closest_position(coordinates)
        samples = np.empty(pos_coords.shape, dtype=float)
        for i, p in enumerate(pos_coords):
            weights, means, covs = self[p]
            selection = np.random.choice(np.arange(len(weights)), p=weights)
            samples[i] = multivariate_normal.rvs(
                mean=means[selection] + pos[i], cov=covs[selection]
            )
        return samples

    def conditioned_probability(self, z, particles):
        """Computes conditioned probabilities p(z|x) using Gaussian Mixtures.

        See :class:`uwb.map.NoiseMap` and :class:`uwb.map.NoiseMapNormal` for more information.
        This is generally to slow for most applications. TODO C++/Numpy reimplementation.
        """
        pos_coords, pos = self.generator.get_closest_position(particles)
        prob = np.zeros((len(z)))
        for i, p in enumerate(pos_coords):
            weights, means, covs = self[p]
            for j, w in enumerate(weights):
                prob[i] += w * multivariate_normal.pdf(
                    z[i], mean=means[j] + pos[i], cov=covs[j]
                )
        return prob

    def __getitem__(self, item):
        """Access to parameters given tuple indices."""
        if len(item) == self._dim:
            param_list = reduce(lambda params, idx: params[idx], item, self.params)
            return param_list[0]
