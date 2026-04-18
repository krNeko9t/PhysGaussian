"""State export helpers for NewtonVBD backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physics_sim.backend.base import SimulationState
from physics_sim.backend.newton_common import (
    compose_body_quat_wxyz,
    pack_cov3x3_to_6,
    quat_xyzw_to_rotmat,
    unpack_cov6_to_3x3,
)
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def export_state(
    *,
    state_0: Any,
    rigid_bodies: list[Any],
    soft_bodies: list[Any],
    n_particles: int,
    device: str,
    init_quats: torch.Tensor | None,
    init_scales: torch.Tensor | None,
    frame_counter: int,
    debug_soft_no_deformation: bool,
    sv_clamp_min: float,
    sv_clamp_max: float,
) -> tuple[SimulationState, int]:
    """Export simulation state and return updated frame counter."""
    positions = torch.zeros((n_particles, 3), device=device, dtype=torch.float32)
    covariances = torch.zeros((n_particles, 6), device=device, dtype=torch.float32)
    rotations = torch.zeros((n_particles, 3, 3), device=device, dtype=torch.float32)

    has_2dgs = init_quats is not None and init_scales is not None
    out_quats = out_scales = None
    if has_2dgs:
        out_quats = torch.zeros((n_particles, 4), device=device, dtype=torch.float32)
        out_scales = torch.zeros(
            (n_particles, init_scales.shape[1]),
            device=device,
            dtype=torch.float32,
        )

    _populate_rigid_state(
        state_0=state_0,
        rigid_bodies=rigid_bodies,
        positions=positions,
        covariances=covariances,
        rotations=rotations,
        has_2dgs=has_2dgs,
        out_quats=out_quats,
        out_scales=out_scales,
        device=device,
    )

    frame_counter += 1
    _populate_soft_state(
        state_0=state_0,
        soft_bodies=soft_bodies,
        positions=positions,
        covariances=covariances,
        rotations=rotations,
        has_2dgs=has_2dgs,
        frame_counter=frame_counter,
        debug_soft_no_deformation=debug_soft_no_deformation,
        sv_clamp_min=sv_clamp_min,
        sv_clamp_max=sv_clamp_max,
        device=device,
    )

    return (
        SimulationState(
            positions=positions,
            covariances=covariances,
            rotations=rotations,
            quats=out_quats if has_2dgs else None,
            scales=out_scales if has_2dgs else None,
        ),
        frame_counter,
    )


def _populate_rigid_state(
    *,
    state_0: Any,
    rigid_bodies: list[Any],
    positions: torch.Tensor,
    covariances: torch.Tensor,
    rotations: torch.Tensor,
    has_2dgs: bool,
    out_quats: torch.Tensor | None,
    out_scales: torch.Tensor | None,
    device: str,
) -> None:
    if not rigid_bodies:
        return

    body_q = state_0.body_q.numpy()
    for body in rigid_bodies:
        t = body_q[body.body_idx]
        body_pos = torch.tensor(t[:3], device=device, dtype=torch.float32)
        body_quat = torch.tensor(t[3:7], device=device, dtype=torch.float32)
        R = quat_xyzw_to_rotmat(body_quat)

        idx = body.particle_indices
        local_pos = body.init_local_pos
        new_pos = (R @ local_pos.T).T + body_pos
        positions[idx] = new_pos

        init_cov = body.init_cov_3x3
        new_cov_3x3 = R @ init_cov @ R.T
        covariances[idx] = pack_cov3x3_to_6(new_cov_3x3)

        R_batch = R.unsqueeze(0).expand(len(idx), -1, -1)
        rotations[idx] = R_batch

        if has_2dgs and body.init_quats is not None:
            out_quats[idx] = compose_body_quat_wxyz(body_quat, body.init_quats)
            out_scales[idx] = body.init_scales


def _populate_soft_state(
    *,
    state_0: Any,
    soft_bodies: list[Any],
    positions: torch.Tensor,
    covariances: torch.Tensor,
    rotations: torch.Tensor,
    has_2dgs: bool,
    frame_counter: int,
    debug_soft_no_deformation: bool,
    sv_clamp_min: float,
    sv_clamp_max: float,
    device: str,
) -> None:
    if not soft_bodies:
        return
    if has_2dgs:
        LOGGER.debug("[NewtonVBD] 2DGS soft export currently keeps rigid-only quats/scales")

    particle_q = state_0.particle_q.numpy()
    do_diag = (frame_counter <= 3) or (frame_counter % 50 == 0)
    for soft in soft_bodies:
        idx = soft.particle_indices
        N = len(idx)

        off = soft.vert_offset
        cur_verts = particle_q[off : off + soft.vert_count]

        if do_diag:
            z_min = cur_verts[:, 2].min()
            z_max = cur_verts[:, 2].max()
            delta = cur_verts - soft.rest_verts
            max_delta = np.abs(delta).max()
            mean_dz = delta[:, 2].mean()
            nan_v = np.isnan(cur_verts).any()
            inf_v = np.isinf(cur_verts).any()
            LOGGER.info(
                "[VBD-DIAG] frame=%s z=[%.4f,%.4f] max_delta=%.6f mean_dz=%.6f nan=%s inf=%s",
                frame_counter,
                z_min,
                z_max,
                max_delta,
                mean_dz,
                nan_v,
                inf_v,
            )

        tet_idx = soft.tet_ids
        bary = soft.bary_coords
        cells = soft.tet_cells
        cell_verts = cells[tet_idx]
        v0 = cur_verts[cell_verts[:, 0]]
        v1 = cur_verts[cell_verts[:, 1]]
        v2 = cur_verts[cell_verts[:, 2]]
        v3 = cur_verts[cell_verts[:, 3]]

        new_pos_np = (
            bary[:, 0:1] * v0
            + bary[:, 1:2] * v1
            + bary[:, 2:3] * v2
            + bary[:, 3:4] * v3
        )
        positions[idx] = torch.from_numpy(new_pos_np.astype(np.float32)).to(device)

        if debug_soft_no_deformation:
            covariances[idx] = soft.init_cov_6.to(device)
            eye = torch.eye(3, device=device, dtype=torch.float32)
            rotations[idx] = eye.unsqueeze(0).expand(N, -1, -1)
            continue

        rest = soft.rest_verts
        rest_cells = cells[tet_idx]
        r0 = rest[rest_cells[:, 0]]
        r1 = rest[rest_cells[:, 1]]
        r2 = rest[rest_cells[:, 2]]
        r3 = rest[rest_cells[:, 3]]

        D_cur = np.stack([v0 - v3, v1 - v3, v2 - v3], axis=-1)
        F_np = np.zeros((N, 3, 3), dtype=np.float32)
        unique_tets = np.unique(tet_idx)
        inv_cache: dict[int, np.ndarray] = {}
        for ut in unique_tets:
            t_cells = cells[ut]
            dr = np.stack(
                [
                    rest[t_cells[0]] - rest[t_cells[3]],
                    rest[t_cells[1]] - rest[t_cells[3]],
                    rest[t_cells[2]] - rest[t_cells[3]],
                ],
                axis=-1,
            )
            try:
                inv_cache[ut] = np.linalg.inv(dr).astype(np.float32)
            except np.linalg.LinAlgError:
                inv_cache[ut] = np.eye(3, dtype=np.float32)

        for i in range(N):
            F_np[i] = D_cur[i] @ inv_cache[tet_idx[i]]
        F_t = torch.from_numpy(F_np).to(device)

        U, S, Vh = torch.linalg.svd(F_t)
        if do_diag:
            s_min = S.min().item()
            s_max = S.max().item()
            det_F = torch.det(F_t)
            n_inv = (det_F < 0).sum().item()
            n_nan = torch.isnan(F_t).any(dim=(1, 2)).sum().item()
            LOGGER.info(
                "[VBD-DIAG] frame=%s sv=[%.4f,%.4f] inverted=%s/%s nan_F=%s",
                frame_counter,
                s_min,
                s_max,
                n_inv,
                N,
                n_nan,
            )

        S_clamped = S.clamp(min=sv_clamp_min, max=sv_clamp_max)
        F_clamped = U @ torch.diag_embed(S_clamped) @ Vh

        init_cov_3x3 = unpack_cov6_to_3x3(soft.init_cov_6.to(device))
        new_cov_3x3 = F_clamped @ init_cov_3x3 @ F_clamped.transpose(1, 2)
        covariances[idx] = pack_cov3x3_to_6(new_cov_3x3)

        R_batch = U @ Vh
        det = torch.det(R_batch)
        mask = det < 0
        if mask.any():
            U_fix = U[mask].clone()
            U_fix[:, :, -1] *= -1
            R_batch[mask] = U_fix @ Vh[mask]
        rotations[idx] = R_batch
