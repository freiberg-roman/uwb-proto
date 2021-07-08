import hydra
import numpy as np
from omegaconf import DictConfig

from uwb.algorithm import BasicParticleFilter, MNMAParticleFilter
from uwb.generator import BlobGenerator, FileMeasurements, RngSensorMeasurements
from uwb.map import NoiseMapGM, NoiseMapNormal


def get_initial_particles():
    """This method needs to be linked to the source of initial particles and weights"""
    p = (
        np.stack([np.arange(10), (np.arange(10) + 1) ** 2, (np.arange(10) + 5) ** 3]).T,
    )
    w = np.ones(10) * 0.1
    return p, w


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.generator.name == "BlobGenerator":
        generator = BlobGenerator(
            grid_dims=cfg.generator.grid_dims,
            step_size=cfg.generator.step_size,
            measurements_per_location=cfg.generator.measurements_per_location,
            modal_range=cfg.generator.modal_range,
            deviation=cfg.generator.deviation,
        )

    if cfg.dynamics.name == "DynamicModel":
        pass
        # model = DynamicModel(std=cfg.dynamics.std)

    if cfg.map.name == "NoiseMapNormal":
        noise_map = NoiseMapNormal(generator=generator)
    elif cfg.map.name == "NoiseMapGM":
        noise_map = NoiseMapGM(
            generator=generator, eps=cfg.map.eps, min_samples=cfg.map.mini_samples
        )
    else:
        raise ValueError("No noise map provided")
    noise_map.gen()

    if cfg.algorithm.name == "BasicParticleFilter":
        pf = BasicParticleFilter(
            *get_initial_particles(),
        )
    elif cfg.algorithm.name == "MNMAParticleFilter":
        pf = MNMAParticleFilter(
            *get_initial_particles(),
            map=noise_map,
        )
    else:
        raise ValueError("No particle filter provided")

    if cfg.measurements.name == "FileMeasurements":
        measurement_generator = FileMeasurements(cfg.measurements.file, 3)
    elif cfg.measurements.name == "RngSensorMeasurements":
        measurement_generator = RngSensorMeasurements(
            cfg.measurements.ranges, cfg.measurements.amount, cfg.measurements.dim
        )
    # main loop
    for i, mb in enumerate(measurement_generator):
        pf.update_weights(mb)

        if i % cfg.resample_each:
            pf.resample()


if __name__ == "__main__":
    run()
