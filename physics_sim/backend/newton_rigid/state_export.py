"""State export helpers for NewtonRigid backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physics_sim.backend.base import SimulationState
from physics_sim.backend.newton_common import (
    compose_body_quat_wxyz,
    pack_cov3x3_to_6,
    quat_xyzw_to_rotmat,
)
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def export_rigid_state(
    *,
    state_0: Any,
    bodies: list[Any],
    n_particles: int,
    device: str,
    init_scales: torch.Tensor | None,
    has_init_quats: bool,
    last_valid_body_q: np.ndarray | None,
) -> tuple[SimulationState, np.ndarray]:
    """Export rigid backend state and return updated fallback poses."""
    body_q = state_0.body_q.numpy()
    corrected = body_q.copy()
    for i in range(corrected.shape[0]):
        if np.any(~np.isfinite(corrected[i])):
            if last_valid_body_q is not None:
                LOGGER.warning(
                    "[NewtonRigid] body=%s pose has NaN/Inf, fallback to last valid pose",
                    i,
                )
                corrected[i] = last_valid_body_q[i]
            else:
                LOGGER.warning(
                    "[NewtonRigid] body=%s pose has NaN/Inf at first export",
                    i,
                )

    positions = torch.zeros((n_particles, 3), device=device, dtype=torch.float32)
    covariances = torch.zeros((n_particles, 6), device=device, dtype=torch.float32)
    rotations = torch.zeros((n_particles, 3, 3), device=device, dtype=torch.float32)

    out_quats = out_scales = None
    if has_init_quats and init_scales is not None:
        out_quats = torch.zeros((n_particles, 4), device=device, dtype=torch.float32)
        out_scales = torch.zeros(
            (n_particles, init_scales.shape[1]),
            device=device,
            dtype=torch.float32,
        )

    for body in bodies:
        t = corrected[body.body_idx]
        body_pos = torch.tensor(t[:3], device=device, dtype=torch.float32)
        body_quat = torch.tensor(t[3:7], device=device, dtype=torch.float32)
        R = quat_xyzw_to_rotmat(body_quat)

        idx = body.particle_indices
        local_pos = body.init_local_pos
        init_cov = body.init_cov_3x3
        new_pos = (R @ local_pos.T).T + body_pos
        R_batch = R.unsqueeze(0)
        new_cov_3x3 = R_batch @ init_cov @ R_batch.transpose(-1, -2)
        R_expand = R.unsqueeze(0).expand(len(idx), -1, -1)

        positions[idx] = new_pos
        covariances[idx] = pack_cov3x3_to_6(new_cov_3x3)
        rotations[idx] = R_expand

        if out_quats is not None and body.init_quats is not None:
            out_quats[idx] = compose_body_quat_wxyz(body_quat, body.init_quats)
            out_scales[idx] = body.init_scales

    return (
        SimulationState(
            positions=positions,
            covariances=covariances,
            rotations=rotations,
            quats=out_quats,
            scales=out_scales,
        ),
        corrected,
    )
