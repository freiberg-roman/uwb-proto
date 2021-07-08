import numpy as np

from uwb.algorithm import BasicParticleFilter


def test_update_weights():
    bpf = BasicParticleFilter(
        np.stack([np.arange(10), (np.arange(10) + 1) ** 2, (np.arange(10) + 5) ** 3]).T,
        np.ones(10) * 0.1,
    )

    bpf.update_weight(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert np.abs(np.sum(bpf.weights) - 1) < 1e-3


def test_resample():
    bpf = BasicParticleFilter(
        np.stack([np.arange(10), (np.arange(10) + 1) ** 2, (np.arange(10) + 5) ** 3]).T,
        np.ones(10) * 0.1,
    )

    bpf.update_weight(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert np.abs(np.sum(bpf.weights) - 1) < 1e-3

    bpf.resample()
    assert np.allclose(bpf.weights, np.ones(10) * 0.1)
