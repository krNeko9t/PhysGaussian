"""Render-only alocasia scene with external default camera view."""

from physics_sim.config.models import (
    CameraConfig,
    NoneBackendConfig,
    ObjectConfig,
    PlySource,
    PreprocessConfig,
    SimConfig,
    TimeConfig,
)

config = SimConfig(
    output="output/alocasia_external_cam",
    backend=NoneBackendConfig(),
    time=TimeConfig(
        substep_dt=1e-3,
        frame_dt=1e-2,
        frame_num=3,
    ),
    preprocess=PreprocessConfig(
        source_up="+Y",
        source_front="+Z",
    ),
    camera=CameraConfig(
        camera_mode="external",
        camera_path="/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/PhysGaussian/datas/physics_dreamer/alocasia/transforms_train.json",
        camera_format="nerfstudio",
        camera_pose_convention="opengl_c2w",
        camera_world_frame="source",
        camera_index=0,
    ),
    objects=[
        ObjectConfig(
            name="alocasia",
            source=PlySource(
                ply_path="datas/physics_dreamer/alocasia/point_cloud.ply",
            ),
        ),
    ],
    boundary_conditions=[],
)
