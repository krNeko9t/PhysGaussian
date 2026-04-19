"""Coordinate transform operations built on SourceAxes."""

from __future__ import annotations

import numpy as np
import torch

from .axes import SourceAxes, alignment_matrix_np
from .camera import (
    align_camera_position as _align_camera_position_impl,
    align_camera_w2c_rotation as _align_camera_w2c_rotation_impl,
)


def align_positions(
    positions: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    if axes.is_identity:
        return positions
    A = axes.A.to(positions.device)
    return positions @ A.T


def inverse_align_positions(
    positions: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    if axes.is_identity:
        return positions
    A_inv = axes.A_inv.to(positions.device)
    return positions @ A_inv.T


def align_directions(dirs: torch.Tensor, axes: SourceAxes) -> torch.Tensor:
    return align_positions(dirs, axes)


def align_covariances(
    cov_upper: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    if axes.is_identity:
        return cov_upper
    A = axes.A.to(cov_upper.device)
    N = cov_upper.shape[0]
    C = torch.zeros((N, 3, 3), device=cov_upper.device, dtype=cov_upper.dtype)
    C[:, 0, 0] = cov_upper[:, 0]
    C[:, 0, 1] = C[:, 1, 0] = cov_upper[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov_upper[:, 2]
    C[:, 1, 1] = cov_upper[:, 3]
    C[:, 1, 2] = C[:, 2, 1] = cov_upper[:, 4]
    C[:, 2, 2] = cov_upper[:, 5]

    C_new = A @ C @ A.T
    return torch.stack(
        [
            C_new[:, 0, 0],
            C_new[:, 0, 1],
            C_new[:, 0, 2],
            C_new[:, 1, 1],
            C_new[:, 1, 2],
            C_new[:, 2, 2],
        ],
        dim=1,
    )


def inverse_align_covariances(
    cov_upper: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    if axes.is_identity:
        return cov_upper
    A_inv = axes.A_inv.to(cov_upper.device)
    N = cov_upper.shape[0]
    C = torch.zeros((N, 3, 3), device=cov_upper.device, dtype=cov_upper.dtype)
    C[:, 0, 0] = cov_upper[:, 0]
    C[:, 0, 1] = C[:, 1, 0] = cov_upper[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov_upper[:, 2]
    C[:, 1, 1] = cov_upper[:, 3]
    C[:, 1, 2] = C[:, 2, 1] = cov_upper[:, 4]
    C[:, 2, 2] = cov_upper[:, 5]

    C_new = A_inv @ C @ A_inv.T
    return torch.stack(
        [
            C_new[:, 0, 0],
            C_new[:, 0, 1],
            C_new[:, 0, 2],
            C_new[:, 1, 1],
            C_new[:, 1, 2],
            C_new[:, 2, 2],
        ],
        dim=1,
    )


def align_quats(quats_wxyz: torch.Tensor, axes: SourceAxes) -> torch.Tensor:
    if axes.is_identity:
        return quats_wxyz
    from physics_sim.rotation_utils import rotmat_to_quat_wxyz, quat_mul_wxyz

    A = axes.A.to(quats_wxyz.device)
    q_align = rotmat_to_quat_wxyz(A.unsqueeze(0)).squeeze(0)
    return quat_mul_wxyz(q_align.unsqueeze(0), quats_wxyz)


def inverse_align_quats(quats_wxyz: torch.Tensor, axes: SourceAxes) -> torch.Tensor:
    if axes.is_identity:
        return quats_wxyz
    from physics_sim.rotation_utils import rotmat_to_quat_wxyz, quat_mul_wxyz

    A_inv = axes.A_inv.to(quats_wxyz.device)
    q_inv = rotmat_to_quat_wxyz(A_inv.unsqueeze(0)).squeeze(0)
    return quat_mul_wxyz(q_inv.unsqueeze(0), quats_wxyz)


def align_camera_w2c_rotation(
    R_w2c: np.ndarray,
    axes: SourceAxes,
) -> np.ndarray:
    A = alignment_matrix_np(axes)
    return _align_camera_w2c_rotation_impl(R_w2c, A)


def align_camera_position(
    position: np.ndarray,
    axes: SourceAxes,
) -> np.ndarray:
    A = alignment_matrix_np(axes)
    return _align_camera_position_impl(position, A)
