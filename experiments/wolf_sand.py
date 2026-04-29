"""Single-object sand simulation (newton_mpm, migrated from warp_mpm)."""

from physics_sim.config.models import (
    BoundingBox,
    CameraConfig,
    FillingConfig,
    MPMMaterial,
    NewtonMPMConfig,
    PlySource,
    PreprocessConfig,
    ReleaseParticlesSequentially,
    SimConfig,
    SurfaceCollider,
    TimeConfig,
)
from physics_sim.config.scene import PartConfig, SceneConfig

config = SimConfig(
    output="output/wolf_sand",
    backend=NewtonMPMConfig(n_grid=64),
    time=TimeConfig(substep_dt=2e-5, frame_dt=4e-2, frame_num=60),
    preprocess=PreprocessConfig(source_up="+Z", source_front="-Y"),
    camera=CameraConfig(),
    boundary_conditions=[
        BoundingBox(),
        SurfaceCollider(
            point=(1, 1, 0.48),
            normal=(0, 0, 1),
            surface="sticky",
            friction=0.0,
        ),
        ReleaseParticlesSequentially(
            normal=(1, 0, 0),
            start_position=1.5,
            end_position=0.5,
            num_layers=50,
            start_time=0.0,
            end_time=1.5,
        ),
    ],
    scene=SceneConfig(
        parts=[
            PartConfig(
                name="wolf",
                source=PlySource(
                    ply_path="model/wolf_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply",
                ),
                # material=MPMMaterial.sand(
                #     density=2000,
                #     E=5e7,
                #     nu=0.3,
                #     friction_angle=30,
                # ),
                material=MPMMaterial.jelly(
                    density=300.0, E=1e6, nu=0.3, yield_stress=1e5,
                ),
                # particle_filling=FillingConfig(
                #     n_grid = 128,
                #     threshold_mode = 'quantile',
                #     density_threshold=0.9,
                #     search_threshold=0.1,
                #     max_particles_num=1_000_000,
                #     max_particles_per_cell=1,
                # ),
            )
        ]
    ),
)
