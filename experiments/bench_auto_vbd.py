"""Auto-generated bench scene with VBD backend."""

from physics_sim.config.models import (
    CameraConfig,
    ColliderConfig,
    NewtonVBDConfig,
    ObjectConfig,
    PlySource,
    PreprocessConfig,
    SimConfig,
    TimeConfig,
)

config = SimConfig(
    output="output/bench_auto_vbd",
    backend=NewtonVBDConfig(),
    time=TimeConfig(frame_num=120),
    preprocess=PreprocessConfig(
        opacity_threshold=0.1,
        axis_permutation="xz-y",
    ),
    camera=CameraConfig(
        width=800, height=600,
        fovx_deg=60.0, fovy_deg=45.0,
        init_azimuth=160.0, init_elevation=20.0, init_radius=2.8,
        delta_a=-0.6, delta_e=0.0,
    ),
    objects=[
        ObjectConfig(
            name="3dovs_bench_inst000",
            source=PlySource(ply_path="scene_data/plys/3dovs_bench_inst000.ply"),
            material=dict(
                density=981.8, mu=0.474, E=44058682800.0, nu=0.313,
                collision_geometry="convex_hull", g=[0, 0, -9.8],
            ),
        ),
        ObjectConfig(
            name="3dovs_bench_inst001",
            source=PlySource(ply_path="scene_data/plys/3dovs_bench_inst001.ply"),
            material=dict(
                density=328.3, mu=0.485, E=828444.0, nu=0.423,
                collision_geometry="convex_hull", g=[0, 0, -9.8],
            ),
        ),
        ObjectConfig(
            name="3dovs_bench_inst003",
            role="collider_only",
            source=PlySource(ply_path="scene_data/plys/3dovs_bench_inst003.ply"),
            collider=ColliderConfig(
                type="plane", space="world",
                surface="slip", friction=0.703,
                fit=dict(method="svd", sample_max=200000, seed=0),
            ),
        ),
    ],
    boundary_conditions=[],
)
