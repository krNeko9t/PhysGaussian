"""Stage 3: Camera setup.

The camera is set up in the **internal Y-up** coordinate system.
For orbit and fixed cameras, the "world up" is always ``[0, 1, 0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from physics_sim.config.models import CameraConfig, SimConfig
from physics_sim.preprocessing.transform import generate_local_coord

if TYPE_CHECKING:
    from physics_sim.stages.scene_setup import SceneData


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
    built to orbit around the scene center with Y as the vertical axis.
    """
    import os
    cam = cfg.camera
    state = CameraState()

    # Determine scene center from available positions (already Y-up)
    if scene_data.sim_objects:
        ref_pos = scene_data.sim_init_pos
    elif scene_data.static_chunks:
        ref_pos = scene_data.static_pos
    else:
        ref_pos = torch.zeros(1, 3, device="cuda")

    lo = torch.min(ref_pos, dim=0)[0]
    hi = torch.max(ref_pos, dim=0)[0]
    scene_center = ((lo + hi) * 0.5).detach().cpu().numpy()

    if cam.camera_mode == "json":
        cameras_json = cam.cameras_json
        if cameras_json is None:
            raise FileNotFoundError(
                "camera_mode='json' requires 'cameras_json' in the config."
            )
        cameras_json = str(cameras_json)
        if not os.path.isabs(cameras_json):
            cameras_json = os.path.join(config_dir, cameras_json)

        # For json cameras, the viewpoint center and up direction are
        # in internal Y-up. The user can override via config if needed.
        state.viewpoint_center_worldspace = scene_center

        # Internal Y-up: vertical is always [0, 1, 0]
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vertical, h1, h2 = generate_local_coord(world_up)
        state.observant_coordinates = np.column_stack((h1, h2, vertical))

    else:
        state.viewpoint_center_worldspace = scene_center

        # Internal Y-up: vertical is always [0, 1, 0]
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vertical, h1, h2 = generate_local_coord(world_up)
        state.observant_coordinates = np.column_stack((h1, h2, vertical))

    cam_dict = cam.model_dump()
    state.camera_params = {k: v for k, v in cam_dict.items() if v is not None}

    return state
