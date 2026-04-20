"""Auto-generated bench scene (IGGT cluster PLYs) with VBD backend."""

from physics_sim.config.models import (
    CameraConfig,
    ColliderConfig,
    ColliderFitSpec,
    NewtonVBDConfig,
    ObjectConfig,
    PlySource,
    PreprocessConfig,
    SimConfig,
    TimeConfig,
    VBDMaterial,
    VBDRigidBody,
)


def _rigid(density: float, mu: float) -> VBDMaterial:
    return VBDMaterial(
        body=VBDRigidBody(density=density, mu=mu, collision_geometry="convex_hull"),
    )


config = SimConfig(
    output="output/test",
    backend=NewtonVBDConfig(
        n_grid=200,
        collision_geometry="convex_hull",
        use_sdf=True,
        solver_iterations=20,
        contact_relaxation=0.5,
    ),
    time=TimeConfig(
        substep_dt=0.0001,
        frame_dt=0.01,
        frame_num=1,
    ),
    preprocess=PreprocessConfig(
        opacity_threshold=0.1,
        source_up="-Y",
        source_front="+Z",
    ),
    camera=CameraConfig(
        width=800,
        height=600,
        fovx_deg=60.0,
        fovy_deg=45.0,
        init_azimuth=180.0,
        init_elevation=20.0,
        init_radius=20.8,
        move_camera=True,
        delta_a=-0.1,
        delta_e=0.0,
        delta_r=0.0,
    ),
    objects=[
        ObjectConfig(
            name="3dovs_bench_inst000",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_000.ply",
            ),
            material=_rigid(density=981.8, mu=0.474),
        ),
        ObjectConfig(
            name="3dovs_bench_inst001",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_001.ply",
            ),
            material=_rigid(density=328.3, mu=0.485),
        ),
        ObjectConfig(
            name="3dovs_bench_inst002",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_002.ply",
            ),
            material=_rigid(density=934.8, mu=0.414),
        ),
        ObjectConfig(
            name="3dovs_bench_inst003",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_003.ply",
            ),
            collider=ColliderConfig(
                type="plane",
                space="world",
                surface="slip",
                friction=0.703,
                fit=ColliderFitSpec(method="svd", sample_max=200000, seed=0),
            ),
            material=_rigid(density=800.0, mu=0.5),
        ),
        ObjectConfig(
            name="3dovs_bench_inst004",
            role="collider_only",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_004.ply",
            ),
            material=_rigid(density=633.4, mu=0.412),
        ),
        ObjectConfig(
            name="3dovs_bench_inst005",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_005.ply",
            ),
            collider=ColliderConfig(
                type="plane",
                space="world",
                surface="slip",
                friction=0.5,
                prefer_up=(0.0186, 0.7519, 0.659),
                fit=ColliderFitSpec(method="svd", sample_max=200000, seed=0),
            ),
            material=_rigid(density=800.0, mu=0.5),
        ),
        ObjectConfig(
            name="3dovs_bench_inst006",
            role="collider_only",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_006.ply",
            ),
            material=_rigid(density=375.6, mu=0.513),
        ),
        ObjectConfig(
            name="3dovs_bench_inst007",
            source=PlySource(
                ply_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_iggt_no_crop/postprocess/seg3d_split/hdbscan_mcs700_ms50/cluster_007.ply",
            ),
            material=_rigid(density=619.7, mu=0.5),
        ),
    ],
    boundary_conditions=[],
)

