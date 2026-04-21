"""VBD: wolf (rigid) + bread (soft body)."""

from physics_sim.config.models import (
    CameraConfig,
    FillingConfig,
    NewtonVBDConfig,
    ObjectTransform,
    PlySource,
    PreprocessConfig,
    SimConfig,
    SurfaceCollider,
    TimeConfig,
    VBDMaterial,
    VBDRigidBody,
    VBDSoftBody,
)
from physics_sim.config.scene import PartConfig, SceneConfig

config = SimConfig(
    output="output/wolf_bread_vbd",
    backend=NewtonVBDConfig(
        soft_contact_ke=100.0,
        debug_soft_no_deformation=True,
        sv_clamp_min=0.1,
        sv_clamp_max=5.0,
        solver_iterations=30,
    ),
    time=TimeConfig(substep_dt=2e-3, frame_num=600),
    preprocess=PreprocessConfig(
        opacity_threshold=0.1, source_up="+Z", source_front="-Y"
    ),
    camera=CameraConfig(
        init_azimuth=90,
        init_elevation=20,
        init_radius=8.0,
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
                material=VBDMaterial(
                    body=VBDRigidBody(
                        density=500, mu=0.5, collision_geometry="convex_hull"
                    ),
                ),
            ),
            PartConfig(
                name="bread",
                source=PlySource(
                    ply_path="model/bread-trained/point_cloud/iteration_30000/point_cloud.ply",
                ),
                transform=ObjectTransform(position=(0.0, 0.0, 1.0)),
                material=VBDMaterial(
                    body=VBDSoftBody(density=300, k_mu=1e5, k_lambda=1e5, k_damp=1e-3),
                ),
                particle_filling=FillingConfig(max_particles_per_cell=16),
            ),
        ]
    ),
)
