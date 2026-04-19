"""Shared rigid-body state export helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physics_sim.backend.newton_common.transforms import (
    compose_body_quat_wxyz,
    pack_cov3x3_to_6,
    quat_xyzw_to_rotmat,
)


def sanitize_body_poses(
    *,
    body_q: np.ndarray,
    last_valid_body_q: np.ndarray | None,
    logger: Any,
    log_prefix: str,
) -> np.ndarray:
    """Replace non-finite body poses with last valid values when available."""
    corrected = body_q.copy()
    for i in range(corrected.shape[0]):
        if np.all(np.isfinite(corrected[i])):
            continue
        if last_valid_body_q is not None:
            logger.warning(
                "[%s] body=%s pose has NaN/Inf, fallback to last valid pose",
                log_prefix,
                i,
            )
            corrected[i] = last_valid_body_q[i]
        else:
            logger.warning(
                "[%s] body=%s pose has NaN/Inf at first export",
                log_prefix,
                i,
            )
    return corrected


def populate_rigid_particles(
    *,
    body_q: np.ndarray,
    bodies: list[Any],
    positions: torch.Tensor,
    covariances: torch.Tensor,
    rotations: torch.Tensor,
    device: str,
    out_quats: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
) -> None:
    """Populate per-particle tensors from rigid body transforms."""
    for body in bodies:
        t = body_q[body.body_idx]
        body_pos = torch.tensor(t[:3], device=device, dtype=torch.float32)
        body_quat = torch.tensor(t[3:7], device=device, dtype=torch.float32)
        rot = quat_xyzw_to_rotmat(body_quat)

        idx = body.particle_indices
        local_pos = body.init_local_pos
        init_cov = body.init_cov_3x3
        new_pos = (rot @ local_pos.T).T + body_pos
        rot_batch = rot.unsqueeze(0).expand(len(idx), -1, -1)
        new_cov_3x3 = rot_batch @ init_cov @ rot_batch.transpose(-1, -2)

        positions[idx] = new_pos
        covariances[idx] = pack_cov3x3_to_6(new_cov_3x3)
        rotations[idx] = rot_batch

        if out_quats is not None and out_scales is not None and body.init_quats is not None:
            out_quats[idx] = compose_body_quat_wxyz(body_quat, body.init_quats)
            out_scales[idx] = body.init_scales

