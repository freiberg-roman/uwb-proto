import hydra
import numpy as np
from omegaconf import DictConfig

from uwb.algorithm import BasicParticleFilter


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    bpf = BasicParticleFilter(
        np.stack([np.arange(10), (np.arange(10) + 1) ** 2, (np.arange(10) + 5) ** 3]).T,
        np.ones(10) * 0.1,
    )

    bpf.update_weight(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert np.abs(np.sum(bpf.weights) - 1) < 1e-3

    bpf.resample()
    assert np.allclose(bpf.weights, np.ones(10) * 0.1)


if __name__ == "__main__":
    run()
