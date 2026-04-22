"""Alocasia scene (MPM variant): branches = MPM jelly, base pinned.

Sibling of ``alocasia_external_cam.py`` (VBD version).  The only changes
are the backend and the ``alocasia_moving`` material.  The scene graph
— parts, ProximitySelector, PinToWorld — is identical, demonstrating
that the same constraint specification runs unchanged across backends.

VBD vs MPM behavioural differences to expect:
  - Branches deform as a material-point continuum with yield.  Jelly
    preset gives soft-elastic-with-high-yield-stress — gentle swing
    stays elastic, aggressive bend picks up plastic deformation.
  - GS particles are MPM particles 1:1.  PinToWorld pins those exact
    GS particles at the base (no tet-vertex over-pinning unlike VBD).
  - Visual quality is sensitive to ``NewtonMPMConfig.n_grid`` /
    ``grid_lim`` and to ``TimeConfig.substep_dt``.  If branches melt
    or blow up, bump n_grid or shrink substep_dt.

NOTE: ``alocasia_foreground_static`` is still render_only; it is used
only as the reference surface for ProximitySelector, not a collider.
Making it a real MPM collider requires Phase F2 (setup_collider).
"""

from physics_sim.config.models import (
    CameraConfig,
    MPMMaterial,
    NewtonMPMConfig,
    NoneBackendConfig,
    PointSelectorSource,
    PreprocessConfig,
    SimConfig,
    TimeConfig,
)
from physics_sim.config.scene import (
    PartConfig,
    PinToWorld,
    ProximitySelector,
    SceneConfig,
)

backend = NoneBackendConfig()
backend = NewtonMPMConfig(
    n_grid=200,
    grid_lim=2.0,
)

config = SimConfig(
    output="output/alocasia_external_cam_mpm",
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
                # Swap to MPMMaterial.foam() / .plasticine() / .jelly()
                # to change branch behaviour.  jelly is the softest elastic
                # preset; foam is slightly stiffer; plasticine deforms
                # permanently when the stress crosses yield.
                material=MPMMaterial.jelly(density=300.0, E=1e5, nu=0.3),
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
