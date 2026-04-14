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


def json_camera(cameras_json: str, **kw) -> CameraConfig:
    defaults = dict(
        camera_mode="json",
        cameras_json=cameras_json,
        default_camera_index=0,
    )
    return CameraConfig(**{**defaults, **kw})
