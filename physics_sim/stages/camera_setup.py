"""Stage 3: Camera setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from physics_sim.config.models import CameraConfig, SimConfig
from physics_sim.preprocessing.transform import (
    generate_local_coord,
    get_center_view_worldspace_and_observant_coordinate,
    transform2origin,
)

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
    """Build camera state from config and scene data."""
    import os
    cam = cfg.camera
    state = CameraState()

    if cam.camera_mode == "json":
        cameras_json = cam.cameras_json
        if cameras_json is None:
            raise FileNotFoundError(
                "camera_mode='json' requires 'cameras_json' in the config."
            )
        cameras_json = str(cameras_json)
        if not os.path.isabs(cameras_json):
            cameras_json = os.path.join(config_dir, cameras_json)

        mpm_vc = torch.tensor(cam.mpm_space_viewpoint_center).reshape(1, 3).cuda()
        mpm_up = torch.tensor(cam.mpm_space_vertical_upward_axis).reshape(1, 3).cuda()

        _, scale_origin, mean_pos = transform2origin(
            scene_data.sim_init_pos, cfg.preprocess.scale,
        )
        state.viewpoint_center_worldspace, state.observant_coordinates = (
            get_center_view_worldspace_and_observant_coordinate(
                mpm_vc, mpm_up,
                scene_data.rotation_matrices, scale_origin, mean_pos,
            )
        )
    else:
        if scene_data.sim_objects:
            ref_pos = scene_data.sim_init_pos
        elif scene_data.static_chunks:
            ref_pos = scene_data.static_pos
        else:
            ref_pos = torch.zeros(1, 3, device="cuda")
        lo = torch.min(ref_pos, dim=0)[0]
        hi = torch.max(ref_pos, dim=0)[0]
        state.viewpoint_center_worldspace = (
            ((lo + hi) * 0.5).detach().cpu().numpy()
        )
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vertical, h1, h2 = generate_local_coord(world_up)
        state.observant_coordinates = np.column_stack((h1, h2, vertical))

    cam_dict = cam.model_dump()
    state.camera_params = {k: v for k, v in cam_dict.items() if v is not None}

    return state
