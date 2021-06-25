import hydra
from omegaconf import DictConfig, OmegaConf

from uwb.generator import BlobGenerator
from uwb.map import NoiseMapGM


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # generate multimodal data for a simple grid
    gen = BlobGenerator(
        [2, 2, 3],
        cfg.generator.step_size,
        cfg.generator.measurements_per_location,
        (cfg.generator.modal_range[0], cfg.generator.modal_range[1]),
    )
    NoiseMapGM(gen)

    for d in gen:
        print(d[0].shape)


if __name__ == "__main__":
    run()
