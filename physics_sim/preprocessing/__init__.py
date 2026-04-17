"""Preprocessing utilities.

.. note::
    Coordinate alignment is handled by :mod:`physics_sim.coord`.
    This package currently exposes optional Taichi-based particle filling
    for MPM-style pipelines.
"""

# Particle filling depends on Taichi; make it an optional import so users can
# still run render-only or rigid-only pipelines without installing Taichi.
try:
    from physics_sim.preprocessing.particle_filling import (
        fill_particles,
        get_particle_volume,
        init_filled_particles,
    )
except ModuleNotFoundError:
    fill_particles = None
    get_particle_volume = None
    init_filled_particles = None
