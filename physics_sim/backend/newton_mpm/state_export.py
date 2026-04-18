"""State export helpers for Newton MPM backend."""

from __future__ import annotations

from typing import Any

import warp as wp
import torch

from physics_sim.backend.base import SimulationState
from physics_sim.backend.newton_mpm.kernels import (
    compute_cov_from_F,
    compute_R_from_F,
    compute_R_quats_scales_from_F,
)


def export_mpm_state(
    *,
    state: Any,
    solver: Any,
    prev_state: Any,
    substep_dt: float,
    frames_dirty: bool,
    n_particles: int,
    init_cov: wp.array,
    out_cov: wp.array,
    out_R: wp.array,
    device: str,
    init_quats_wp: wp.array | None,
    init_scales_wp: wp.array | None,
    out_quats_wp: wp.array | None,
    out_scales_wp: wp.array | None,
    num_scales: int,
) -> tuple[SimulationState, bool]:
    """Export MPM simulation state and return updated dirty-flag."""
    if frames_dirty:
        solver.update_particle_frames(prev_state, state, substep_dt)
        frames_dirty = False

    pos = wp.to_torch(state.particle_q)
    vel = wp.to_torch(state.particle_qd)

    F = state.mpm.particle_transform
    wp.launch(
        compute_cov_from_F,
        dim=n_particles,
        inputs=[F, init_cov, out_cov],
        device=device,
    )
    cov = wp.to_torch(out_cov).view(n_particles, 6)

    out_quats_t = None
    out_scales_t = None
    if init_quats_wp is not None:
        wp.launch(
            compute_R_quats_scales_from_F,
            dim=n_particles,
            inputs=[
                F,
                init_quats_wp,
                init_scales_wp,
                num_scales,
                out_R,
                out_quats_wp,
                out_scales_wp,
            ],
            device=device,
        )
        out_quats_t = wp.to_torch(out_quats_wp).view(n_particles, 4)
        out_scales_t = wp.to_torch(out_scales_wp).view(n_particles, num_scales)
    else:
        wp.launch(
            compute_R_from_F,
            dim=n_particles,
            inputs=[F, out_R],
            device=device,
        )
    rot = wp.to_torch(out_R).view(n_particles, 3, 3)
    return (
        SimulationState(
            positions=pos,
            covariances=cov,
            rotations=rot,
            velocities=vel,
            quats=out_quats_t,
            scales=out_scales_t,
        ),
        frames_dirty,
    )
