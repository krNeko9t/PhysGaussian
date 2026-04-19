"""Camera configuration preset factories."""

from physics_sim.config.models import CameraConfig


def orbit_camera(**kw) -> CameraConfig:
    defaults = dict(camera_mode="orbit")
    return CameraConfig(**{**defaults, **kw})


def fixed_camera(**kw) -> CameraConfig:
    defaults = dict(
        camera_mode="fixed",
        width=800,
        height=600,
        fovx_deg=60.0,
        fovy_deg=45.0,
    )
    return CameraConfig(**{**defaults, **kw})


def external_camera(
    camera_path: str,
    *,
    camera_format: str,
    camera_pose_convention: str,
    camera_world_frame: str = "source",
    camera_index: int = 0,
    **kw,
) -> CameraConfig:
    defaults = dict(
        camera_mode="external",
        camera_path=camera_path,
        camera_format=camera_format,
        camera_pose_convention=camera_pose_convention,
        camera_world_frame=camera_world_frame,
        camera_index=camera_index,
    )
    return CameraConfig(**{**defaults, **kw})
