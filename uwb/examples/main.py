import hydra
from omegaconf import DictConfig, OmegaConf

from uwb.generator import BlobGenerator


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    gen = BlobGenerator(100, 200, 10, 50, (1, 3))
    gen.gen()


if __name__ == "__main__":
    run()
