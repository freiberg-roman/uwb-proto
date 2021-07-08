import numpy as np

from uwb.algorithm import MNMAParticleFilter
from uwb.generator import BlobGenerator
from uwb.map import NoiseMapGM, NoiseMapNormal


def test_update_weights():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 1),
        deviation=1.0,
    )
    noise_map = NoiseMapGM(generator=bg)
    noise_map.gen()

    mnmapf = MNMAParticleFilter(
        np.stack([np.arange(10), (np.arange(10) + 1) ** 2, (np.arange(10) + 5) ** 3]).T,
        np.ones(10) * 0.1,
        map=noise_map,
    )

    mnmapf.update_weights(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert np.abs(np.sum(mnmapf.weights) - 1) < 1e-3


def test_resample():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 5),
        deviation=1.0,
    )
    noise_map = NoiseMapNormal(generator=bg)
    noise_map.gen()

    mnmapf = MNMAParticleFilter(
        np.stack([np.arange(10), (np.arange(10) + 1) ** 2, (np.arange(10) + 5) ** 3]).T,
        np.ones(10) * 0.1,
        map=noise_map,
    )

    mnmapf.update_weights(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert np.abs(np.sum(mnmapf.weights) - 1) < 1e-3

    mnmapf.resample()
    assert np.allclose(mnmapf.weights, np.ones(10) * 0.1)
    assert len(mnmapf.particles) == 10
