import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from uwb.generator import BlobGenerator
from uwb.map import NoiseMapGM, NoiseMapNormal


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(0)
    # generate multimodal data for a simple grid
    gen = BlobGenerator(
        [2, 2, 3],
        cfg.generator.step_size,
        cfg.generator.measurements_per_location,
        (cfg.generator.modal_range[0], cfg.generator.modal_range[1]),
    )
    gm_nm = NoiseMapGM(gen)
    n_nm = NoiseMapNormal(gen)

    gm_nm.gen()
    n_nm.gen()
    w, m, c = gm_nm[(0, 0, 0)]
    print(w, m, c)


if __name__ == "__main__":
    run()
