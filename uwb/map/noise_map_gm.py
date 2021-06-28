from copy import deepcopy
from functools import reduce

import numpy as np
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

            for i in range(n_clusters):
                mask = labels == i
                mean_noise = samples[mask, :].mean(axis=0) - pos
                cov_noise = np.cov(samples[mask, :].T)
                weight = mask.sum() / used_data
                param_list.append((mean_noise, cov_noise, weight, pos))
