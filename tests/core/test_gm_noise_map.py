import numpy as np

from uwb.generator import BlobGenerator
from uwb.map import NoiseMapGM


def test_generate_normal_noise_map():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 1),
        deviation=1.0,
    )
    noise_map = NoiseMapGM(generator=bg)
    noise_map.gen()
    assert len(noise_map[(0, 0, 0)]) == 3  # should have weights, means, covariances


def test_conditioned_prob():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 5),
        deviation=1.0,
    )
    noise_map = NoiseMapGM(generator=bg)
    noise_map.gen()

    prob = noise_map.conditioned_probability(
        np.array([[1.0, 1.0, 1.0]]), np.array([[12.0, 13.0, 14.0]])
    )
    assert prob.shape == (1,)
    prob = noise_map.conditioned_probability(
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        np.array([[12.0, 13.0, 14.0], [20.0, 20.0, 21.0]]),
    )
    assert prob.shape == (2,)


def test_samples_from():
    bg = BlobGenerator(
        grid_dims=[2, 4, 6],
        step_size=10,
        measurements_per_location=100,
        modal_range=(1, 5),
        deviation=1.0,
    )

    noise_map = NoiseMapGM(generator=bg)
    noise_map.gen()

    samples = noise_map.sample_from(np.array([[1.0, 1.0, 1.0]]))
    assert samples.shape == (1, 3)

    samples = noise_map.sample_from(np.array([[1.0, 1.0, 1.0], [20.0, 20.0, 20.0]]))
    assert samples.shape == (2, 3)
