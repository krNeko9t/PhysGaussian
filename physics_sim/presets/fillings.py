"""Particle filling configuration preset factories."""

from physics_sim.config.models import FillingConfig


def default_filling(**kw) -> FillingConfig:
    return FillingConfig(**kw)


def dense_filling(**kw) -> FillingConfig:
    defaults = dict(
        max_particles_per_cell=16,
        density_threshold=100.0,
        search_threshold=1.0,
        max_particles_num=2_000_000,
    )
    return FillingConfig(**{**defaults, **kw})
