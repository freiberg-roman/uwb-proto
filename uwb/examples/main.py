import hydra
from omegaconf import DictConfig, OmegaConf

from uwb.generator import BlobGenerator
from uwb.map.noise_map import NoiseMap


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # generate multimodal data for a simple grid
    gen = BlobGenerator(
        cfg.generator.grid_length,
        cfg.generator.grid_width,
        cfg.generator.step_size,
        cfg.generator.measurements_per_location,
        (cfg.generator.modal_range[0], cfg.generator.modal_range[1]),
    )
    map = NoiseMap()
    data = gen.gen()
    map = map.gen(data)
    print(data.shape)


if __name__ == "__main__":
    run()
