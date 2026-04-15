"""Rigid body: wolf + bread dropping onto a ground plane."""

from physics_sim.config.models import (
    NewtonRigidConfig,
    ObjectConfig,
    ObjectTransform,
    PlySource,
    PreprocessConfig,
    SimConfig,
    SurfaceCollider,
    TimeConfig,
    CameraConfig,
)

config = SimConfig(
    output="output/wolf_bread_rigid",
    backend=NewtonRigidConfig(),
    time=TimeConfig(substep_dt=1e-3),
    preprocess=PreprocessConfig(source_up="Z_UP"),
    camera=CameraConfig(
        init_azimuth=90, init_elevation=20, init_radius=3.0,
        delta_a=-1.0, delta_e=0.0,
    ),
    objects=[
        ObjectConfig(
            name="wolf",
            source=PlySource(
                ply_path="model/wolf_whitebg-trained/point_cloud/iteration_30000/point_cloud.ply",
            ),
            material=dict(density=500, mu=0.6, E=1e5, nu=0.3, g=[0, 0, -9.8]),
        ),
        ObjectConfig(
            name="bread",
            source=PlySource(
                ply_path="model/bread-trained/point_cloud/iteration_30000/point_cloud.ply",
            ),
            transform=ObjectTransform(position=(0.0, 0.0, 1.5)),
            material=dict(density=500, mu=0.3, E=1e5, nu=0.3, g=[0, 0, -9.8]),
        ),
    ],
    boundary_conditions=[
        SurfaceCollider(
            point=(1, 1, 0.20), normal=(0, 0, 1),
            surface="sticky", friction=0.6,
        ),
    ],
)
