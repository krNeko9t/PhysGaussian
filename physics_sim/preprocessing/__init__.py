from physics_sim.preprocessing.transform import (
    transform2origin,
    undotransform2origin,
    shift2center111,
    undoshift2center111,
    generate_rotation_matrices,
    apply_rotations,
    apply_inverse_rotations,
    apply_cov_rotations,
    apply_inverse_cov_rotations,
    undo_all_transforms,
)

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
