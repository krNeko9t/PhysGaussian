"""Quaternion and axis-permutation helpers.

.. deprecated::
    The canonical implementations now live in :mod:`physics_sim.rotation_utils`
    and :mod:`physics_sim.coord`.  This module re-exports them for backward
    compatibility with external scripts that import from here.
"""

from __future__ import annotations

import torch

# Re-export canonical implementations
from physics_sim.rotation_utils import quat_mul_wxyz, rotmat_to_quat_wxyz  # noqa: F401


def build_axis_perm_matrix(perm: str, device: torch.device | str) -> torch.Tensor:
    """Return a 3x3 signed permutation matrix for axis string *perm*.

    .. deprecated:: Use ``physics_sim.coord.alignment_matrix`` instead.
    """
    if perm == "xyz":
        return torch.eye(3, device=device, dtype=torch.float32)
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx: list[int] = []
    signs: list[float] = []
    negate_next = False
    for c in perm.lower():
        if c == "-":
            negate_next = True
        elif c in axis_map:
            idx.append(axis_map[c])
            signs.append(-1.0 if negate_next else 1.0)
            negate_next = False
    P = torch.zeros(3, 3, device=device, dtype=torch.float32)
    for row, (col, s) in enumerate(zip(idx, signs)):
        P[row, col] = s
    return P


def apply_axis_perm_to_quats(quats_wxyz: torch.Tensor, perm: str) -> torch.Tensor:
    """Apply axis permutation to an (N, 4) wxyz quaternion tensor.

    .. deprecated:: Use ``physics_sim.coord.align_quats`` instead.
    """
    if perm == "xyz":
        return quats_wxyz
    device = quats_wxyz.device
    P = build_axis_perm_matrix(perm, device)
    if torch.det(P) < 0:
        P = -P
    q_perm = rotmat_to_quat_wxyz(P)
    return quat_mul_wxyz(q_perm, quats_wxyz)


def preprocess_quats(
    quats_wxyz: torch.Tensor,
    axis_perm: str,
    rotation_matrices: list[torch.Tensor],
) -> torch.Tensor:
    """Transform quaternions from world space into backend space.

    .. deprecated:: Use ``physics_sim.coord.align_quats`` instead.
    """
    device = quats_wxyz.device
    R_combined = build_axis_perm_matrix(axis_perm, device)
    if torch.det(R_combined) < 0:
        R_combined = -R_combined
    for R in rotation_matrices:
        R_combined = R @ R_combined
    q_pre = rotmat_to_quat_wxyz(R_combined)
    return quat_mul_wxyz(q_pre, quats_wxyz)


def inverse_preprocess_quats(
    quats_mpm_wxyz: torch.Tensor,
    axis_perm: str,
    rotation_matrices: list[torch.Tensor],
) -> torch.Tensor:
    """Transform quaternions from backend space back to world space.

    .. deprecated:: Use ``physics_sim.coord.inverse_align_quats`` instead.
    """
    device = quats_mpm_wxyz.device
    R_combined = build_axis_perm_matrix(axis_perm, device)
    if torch.det(R_combined) < 0:
        R_combined = -R_combined
    for R in rotation_matrices:
        R_combined = R @ R_combined
    q_pre = rotmat_to_quat_wxyz(R_combined)
    q_pre_inv = q_pre.clone()
    q_pre_inv[1:] = -q_pre_inv[1:]
    return quat_mul_wxyz(q_pre_inv, quats_mpm_wxyz)
