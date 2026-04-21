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
    backend=NewtonMPMConfig(n_grid=200),
    time=TimeConfig(substep_dt=2e-5, frame_dt=4e-2, frame_num=50),
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
                material=MPMMaterial.sand(
                    density=2000,
                    E=5e7,
                    nu=0.3,
                    friction_angle=30,
                ),
                particle_filling=FillingConfig(
                    density_threshold=100.0,
                    search_threshold=1.0,
                    search_exclude_direction=2,
                    max_particles_num=2_000_000,
                    max_particles_per_cell=1,
                    boundary=[0.4, 1.6, 0.4, 1.6, 0.4, 1.6],
                    visualize=True,
                ),
            )
        ]
    ),
)
