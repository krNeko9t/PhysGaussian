"""State export helpers for NewtonRigid backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physics_sim.backend.base import SimulationState
from physics_sim.backend.newton_common import (
    populate_rigid_particles,
    sanitize_body_poses,
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
    corrected = sanitize_body_poses(
        body_q=state_0.body_q.numpy(),
        last_valid_body_q=last_valid_body_q,
        logger=LOGGER,
        log_prefix="NewtonRigid",
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

    populate_rigid_particles(
        body_q=corrected,
        bodies=bodies,
        positions=positions,
        covariances=covariances,
        rotations=rotations,
        device=device,
        out_quats=out_quats,
        out_scales=out_scales,
    )

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
