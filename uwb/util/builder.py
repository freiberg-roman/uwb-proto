from uwb.algorithm import BasicParticleFilter, DynamicModel, MNMAParticleFilter
from uwb.generator import BlobGenerator
from uwb.map import NoiseMapGM, NoiseMapNormal


def create_blob_gen(cfg):
    assert cfg.generator.name == "BlobGenerator"

    return BlobGenerator(
        grid_dims=cfg.generator.grid_dims,
        step_size=cfg.generator.step_size,
        measurements_per_location=cfg.generator.measurements_per_location,
        modal_range=cfg.generator.modal_range,
        deviation=cfg.generator.deviation,
    )


def create_normal_noise_map(cfg, gen):
    assert cfg.map.name == "NoiseMapNormal"

    return NoiseMapNormal(
        generator=gen,
    )


def create_gm_noise_map(cfg, gen):
    assert cfg.map.name == "NoiseMapGM"

    return NoiseMapGM(
        generator=gen,
        eps=cfg.map.eps,
        min_samples=cfg.map.min_samples,
    )


def create_basic_pf(cfg, init_particles, init_weights):
    assert cfg.algorithm.name == "BasicParticleFilter"

    return BasicParticleFilter(
        init_particles=init_particles,
        init_weights=init_weights,
    )


def create_mnmapf(cfg, init_particles, init_weights, map):
    assert cfg.algorithm.name == "MNMAParticleFilter"

    return MNMAParticleFilter(
        init_particles=init_particles,
        init_weights=init_weights,
        map=map,
    )


def create_dyn_model(cfg):
    assert cfg.dynamics.name == "DynamicModel"

    return DynamicModel(std=cfg.dynamics.std)
