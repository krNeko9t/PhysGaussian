"""Alocasia scene: branches = VBD soft body, base pinned to the pot.

Validates the new scene-graph PinToWorld constraint end-to-end:
the ``alocasia_moving`` part (branches + leaves) is a soft body,
and particles whose initial position lies within ``max_distance`` of
``alocasia_foreground_static`` (pot + trunk base) are frozen via
``particle_mass=0`` in VBD's apply_constraints.

Tuning knobs:
  - ``PinToWorld.particles.max_distance``: radius (in Y-up internal
    coords) of the "glued-to-pot" shell.  Too small → nothing gets
    pinned and the whole plant drops; too big → the entire plant is
    frozen.  Start at 0.02 and adjust after watching one run.
  - ``VBDSoftBody`` stiffness: higher ``k_mu``/``k_lambda`` = stiffer
    branches; ``k_damp`` damps oscillation.
"""

from physics_sim.config.models import (
    CameraConfig,
    NewtonVBDConfig,
    NoneBackendConfig,
    PointSelectorSource,
    PreprocessConfig,
    SimConfig,
    TimeConfig,
    VBDMaterial,
    VBDSoftBody,
)
from physics_sim.config.scene import (
    PartConfig,
    PinToWorld,
    ProximitySelector,
    SceneConfig,
)

backend = NoneBackendConfig()
backend = NewtonVBDConfig(
    soft_contact_ke=100.0,
    debug_soft_no_deformation=False,
    sv_clamp_min=0.1,
    sv_clamp_max=5.0,
    solver_iterations=30,
)

config = SimConfig(
    output="output/alocasia_pintoworld_test",
    backend=backend,
    time=TimeConfig(
        substep_dt=1e-3,
        frame_dt=1e-2,
        frame_num=60,
    ),
    preprocess=PreprocessConfig(
        source_up="+Z",
        source_front="+Y",
    ),
    camera=CameraConfig(
        camera_mode="external",
        camera_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/PhysGaussian/datas/physics_dreamer/alocasia/transforms_train.json",
        camera_format="nerfstudio",
        camera_pose_convention="opengl_c2w",
        camera_world_frame="source",
        camera_index=0,
    ),
    boundary_conditions=[],
    scene=SceneConfig(
        parts=[
            PartConfig(
                name="alocasia_moving",
                source=PointSelectorSource(
                    base_ply_path="datas/physics_dreamer/alocasia/point_cloud.ply",
                    selector_path="datas/physics_dreamer/alocasia/moving_part_points.ply",
                    selector_kind="point_cloud_xyz",
                    match_tolerance=1e-5,
                    min_match_ratio=0.95,
                    max_ambiguous_ratio=1e-3,
                ),
                material=VBDMaterial(
                    body=VBDSoftBody(
                        density=300,
                        k_mu=1e5,
                        k_lambda=1e5,
                        k_damp=1e-3,
                    ),
                ),
            ),
            PartConfig(
                name="alocasia_foreground_static",
                source=PointSelectorSource(
                    base_ply_path="datas/physics_dreamer/alocasia/point_cloud.ply",
                    selector_path="datas/physics_dreamer/alocasia/clean_object_points.ply",
                    selector_kind="point_cloud_xyz",
                    subtract_selector_path="datas/physics_dreamer/alocasia/moving_part_points.ply",
                    subtract_selector_kind="point_cloud_xyz",
                    match_tolerance=1e-5,
                    min_match_ratio=0.95,
                    max_ambiguous_ratio=1e-3,
                ),
            ),
            PartConfig(
                name="alocasia_background",
                source=PointSelectorSource(
                    base_ply_path="datas/physics_dreamer/alocasia/point_cloud.ply",
                    selector_path="datas/physics_dreamer/alocasia/clean_object_points.ply",
                    selector_kind="point_cloud_xyz",
                    invert=True,
                    match_tolerance=1e-5,
                    min_match_ratio=0.95,
                    max_ambiguous_ratio=1e-3,
                ),
            ),
        ],
        constraints=[
            PinToWorld(
                particles=ProximitySelector(
                    part="alocasia_moving",
                    to_surface_of="alocasia_foreground_static",
                    max_distance=0.02,
                ),
            ),
        ],
    ),
)

