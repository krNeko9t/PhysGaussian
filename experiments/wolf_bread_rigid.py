"""Rigid body: wolf + bread dropping onto a ground plane."""

from physics_sim.config.models import (
    CameraConfig,
    NewtonRigidConfig,
    ObjectTransform,
    PlySource,
    PreprocessConfig,
    RigidMaterial,
    SimConfig,
    SurfaceCollider,
    TimeConfig,
)
from physics_sim.config.scene import PartConfig, SceneConfig

config = SimConfig(
    output="output/wolf_bread_rigid",
    backend=NewtonRigidConfig(),
    time=TimeConfig(substep_dt=1e-3),
    preprocess=PreprocessConfig(source_up="+Z", source_front="-Y"),
    camera=CameraConfig(
        init_azimuth=90,
        init_elevation=20,
        init_radius=3.0,
        delta_a=-1.0,
        delta_e=0.0,
    ),
    boundary_conditions=[
        SurfaceCollider(
            point=(1, 1, 0.20),
            normal=(0, 0, 1),
            surface="sticky",
            friction=0.6,
        ),
    ],
    scene=SceneConfig(
        parts=[
            PartConfig(
                name="wolf",
                source=PlySource(
                    ply_path="model/wolf_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply",
                ),
                material=RigidMaterial(density=500, mu=0.6),
            ),
            PartConfig(
                name="bread",
                source=PlySource(
                    ply_path="model/bread-trained/point_cloud/iteration_30000/point_cloud.ply",
                ),
                transform=ObjectTransform(position=(0.0, 0.0, 1.5)),
                material=RigidMaterial(density=500, mu=0.3),
            ),
        ]
    ),
)
