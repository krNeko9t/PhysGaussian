"""Stage 3: Camera setup.

The camera is set up in the **internal Y-up** coordinate system.
For orbit cameras, the standard basis is:

    azimuth=0  -> camera at +Z (facing the scene front)
    azimuth=90 -> camera at -X (scene's left side)
    vertical   -> +Y (up)

This gives azimuth a clear semantic meaning once the user configures
``source_front`` correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from physics_sim.config.models import CameraConfig, SimConfig

if TYPE_CHECKING:
    from physics_sim.stages.scene_setup import SceneData


# Standard orbit basis in internal Y-up space.
# h1 = +Z (toward viewer) -> azimuth=0 is front view
# h2 = -X                 -> azimuth=90 is left side view
# vertical = +Y           -> elevation goes up
_ORBIT_H1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_ORBIT_H2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
_ORBIT_VERTICAL = np.array([0.0, 1.0, 0.0], dtype=np.float32)
STANDARD_OBSERVANT_COORDINATES = np.column_stack((_ORBIT_H1, _ORBIT_H2, _ORBIT_VERTICAL))


@dataclass
class CameraState:
    viewpoint_center_worldspace: Any = None
    observant_coordinates: Any = None
    camera_params: dict = field(default_factory=dict)


def setup_camera(
    cfg: SimConfig,
    scene_data: SceneData,
    config_dir: str = "",
) -> CameraState:
    """Build camera state from config and scene data.

    All scene positions are already in internal Y-up. The camera is
    built to orbit around the scene center with a deterministic basis
    so that azimuth=0 corresponds to the "front" view (+Z direction).
    """
    import os
    cam = cfg.camera
    state = CameraState()

    if scene_data.sim_objects:
        ref_pos = scene_data.dynamic_init.pos
    elif scene_data.static_render is not None:
        ref_pos = scene_data.static_render.pos
    else:
        ref_pos = torch.zeros(1, 3, device="cuda")

    lo = torch.min(ref_pos, dim=0)[0]
    hi = torch.max(ref_pos, dim=0)[0]
    scene_center = ((lo + hi) * 0.5).detach().cpu().numpy()

    state.viewpoint_center_worldspace = scene_center
    state.observant_coordinates = STANDARD_OBSERVANT_COORDINATES

    cam_dict = cam.model_dump()
    if cam.camera_mode == "external":
        required = (
            "camera_path",
            "camera_format",
            "camera_pose_convention",
            "camera_world_frame",
        )
        missing = [name for name in required if getattr(cam, name) in (None, "")]
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                "camera_mode='external' requires fields: "
                f"{missing_fields}"
            )
        camera_path = str(cam.camera_path)
        if not os.path.isabs(camera_path):
            camera_path = os.path.join(config_dir, camera_path)
        cam_dict["camera_path"] = os.path.abspath(camera_path)

    state.camera_params = {k: v for k, v in cam_dict.items() if v is not None}

    return state
