"""Quaternion and axis-permutation helpers for 2DGS preprocessing.

Used in the render loop to convert between world-space and
backend-space quaternion orientations.
"""

from __future__ import annotations

import torch


def build_axis_perm_matrix(perm: str, device: torch.device) -> torch.Tensor:
    """Return a 3x3 signed permutation matrix for axis string *perm*."""
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


def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix(ces) to wxyz quaternion(s).

    *R*: ``(3, 3)`` or ``(N, 3, 3)``
    Returns: ``(4,)`` or ``(N, 4)``
    """
    single = R.dim() == 2
    if single:
        R = R.unsqueeze(0)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)
    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(trace[m1] + 1.0) * 2.0
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s
    m2 = ~m1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = torch.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2.0
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s
    m3 = ~m1 & ~m2 & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = torch.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2.0
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s
    m4 = ~m1 & ~m2 & ~m3
    if m4.any():
        s = torch.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]) * 2.0
        q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 3] = 0.25 * s
    q = torch.nn.functional.normalize(q, dim=-1)
    return q[0] if single else q


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions. Broadcasts *a* over *b*."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    aw, ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bw, bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return torch.cat([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def apply_axis_perm_to_quats(quats_wxyz: torch.Tensor, perm: str) -> torch.Tensor:
    """Apply axis permutation to an (N, 4) wxyz quaternion tensor."""
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
    """Transform quaternions from world space into backend space."""
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
    """Transform quaternions from backend space back to world space."""
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
