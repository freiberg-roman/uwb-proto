import hydra
import numpy as np
from omegaconf import DictConfig

from uwb.generator import BlobGenerator
from uwb.map import NoiseMapGM


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
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


if __name__ == "__main__":
    run()
