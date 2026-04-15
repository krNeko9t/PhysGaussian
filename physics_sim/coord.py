"""
Coordinate system conventions and conversions.

This module is the **single source of truth** for all coordinate-system
transformations in this repository.

Internal convention (used everywhere after ingestion):

    Right-handed, **Y-up**
        +X = right
        +Y = up
        +Z = toward viewer  (OpenGL style)

Supported source conventions:

    Y_UP  -- same as internal; identity alignment.
    Z_UP  -- right-handed, Z points up (Blender world / engineering).

All functions operate on **torch** tensors to match the rest of the
pipeline.  Lightweight numpy helpers are provided for camera math.
"""

from __future__ import annotations

import enum
from typing import Optional

import numpy as np
import torch


# ── Enum ──────────────────────────────────────────────────────────────

class UpAxis(enum.Enum):
    Y_UP = "Y_UP"
    Z_UP = "Z_UP"

    @classmethod
    def from_string(cls, s: str) -> "UpAxis":
        """Accept flexible inputs: 'Y_UP', '+Y', 'y', 'Z_UP', '+Z', …"""
        s = s.strip().upper().replace("+", "").replace("-", "")
        if s in ("Y", "Y_UP", "YUP"):
            return cls.Y_UP
        if s in ("Z", "Z_UP", "ZUP"):
            return cls.Z_UP
        raise ValueError(f"Unknown up-axis string: '{s}'")


# ── Core alignment matrices ──────────────────────────────────────────

def _z_up_to_y_up_matrix() -> torch.Tensor:
    """3x3 rotation that maps Z-up right-hand -> Y-up right-hand.

    Rotation of -90 deg around the X axis::

        x ->  x
        y -> -z
        z ->  y
    """
    return torch.tensor([
        [1.0, 0.0,  0.0],
        [0.0, 0.0,  1.0],
        [0.0, -1.0, 0.0],
    ], dtype=torch.float32)


def alignment_matrix(source: UpAxis, device: str = "cuda") -> torch.Tensor:
    """Return 3x3 ``A`` such that ``pos_internal = (A @ pos_source.T).T``.

    Identity when *source* is already Y-up.
    """
    if source is UpAxis.Y_UP:
        return torch.eye(3, dtype=torch.float32, device=device)
    if source is UpAxis.Z_UP:
        return _z_up_to_y_up_matrix().to(device)
    raise ValueError(f"Unsupported source convention: {source}")


def alignment_matrix_np(source: UpAxis) -> np.ndarray:
    """Numpy variant of :func:`alignment_matrix` (for camera math)."""
    return alignment_matrix(source, device="cpu").numpy()


def inverse_alignment_matrix(source: UpAxis, device: str = "cuda") -> torch.Tensor:
    """``A_inv`` such that ``pos_source = (A_inv @ pos_internal.T).T``.

    For orthogonal matrices, ``A_inv == A.T``.
    """
    return alignment_matrix(source, device).T


# ── Batch position / direction helpers ────────────────────────────────

def align_positions(
    positions: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform (N, 3) positions from *source* to internal Y-up."""
    A = alignment_matrix(source, positions.device)
    if source is UpAxis.Y_UP:
        return positions
    return positions @ A.T


def inverse_align_positions(
    positions: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform (N, 3) positions from internal Y-up back to *source*."""
    A_inv = inverse_alignment_matrix(source, positions.device)
    if source is UpAxis.Y_UP:
        return positions
    return positions @ A_inv.T


def align_directions(
    dirs: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform direction vectors (normals, gravity, …) from source to internal."""
    return align_positions(dirs, source)


# ── Covariance alignment ─────────────────────────────────────────────

def align_covariances(
    cov_upper: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform (N, 6) upper-triangle covariances from *source* to internal.

    Upper-triangle layout: [c00, c01, c02, c11, c12, c22].
    Cov transforms as ``C' = A @ C @ A^T``.
    """
    if source is UpAxis.Y_UP:
        return cov_upper
    A = alignment_matrix(source, cov_upper.device)
    N = cov_upper.shape[0]
    C = torch.zeros((N, 3, 3), device=cov_upper.device, dtype=cov_upper.dtype)
    C[:, 0, 0] = cov_upper[:, 0]
    C[:, 0, 1] = C[:, 1, 0] = cov_upper[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov_upper[:, 2]
    C[:, 1, 1] = cov_upper[:, 3]
    C[:, 1, 2] = C[:, 2, 1] = cov_upper[:, 4]
    C[:, 2, 2] = cov_upper[:, 5]

    C_new = A @ C @ A.T
    return torch.stack([
        C_new[:, 0, 0], C_new[:, 0, 1], C_new[:, 0, 2],
        C_new[:, 1, 1], C_new[:, 1, 2], C_new[:, 2, 2],
    ], dim=1)


def inverse_align_covariances(
    cov_upper: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform (N, 6) covariances from internal Y-up back to *source*."""
    if source is UpAxis.Y_UP:
        return cov_upper
    A_inv = inverse_alignment_matrix(source, cov_upper.device)
    N = cov_upper.shape[0]
    C = torch.zeros((N, 3, 3), device=cov_upper.device, dtype=cov_upper.dtype)
    C[:, 0, 0] = cov_upper[:, 0]
    C[:, 0, 1] = C[:, 1, 0] = cov_upper[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov_upper[:, 2]
    C[:, 1, 1] = cov_upper[:, 3]
    C[:, 1, 2] = C[:, 2, 1] = cov_upper[:, 4]
    C[:, 2, 2] = cov_upper[:, 5]

    C_new = A_inv @ C @ A_inv.T
    return torch.stack([
        C_new[:, 0, 0], C_new[:, 0, 1], C_new[:, 0, 2],
        C_new[:, 1, 1], C_new[:, 1, 2], C_new[:, 2, 2],
    ], dim=1)


# ── Quaternion alignment ─────────────────────────────────────────────

def align_quats(
    quats_wxyz: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform (N, 4) wxyz quaternions from *source* to internal Y-up.

    Each Gaussian's orientation is pre-multiplied by the alignment
    quaternion: ``q_internal = q_align * q_source``.
    """
    if source is UpAxis.Y_UP:
        return quats_wxyz
    from physics_sim.rotation_utils import rotmat_to_quat_wxyz, quat_mul_wxyz
    A = alignment_matrix(source, quats_wxyz.device)
    q_align = rotmat_to_quat_wxyz(A.unsqueeze(0)).squeeze(0)  # (4,)
    return quat_mul_wxyz(q_align.unsqueeze(0), quats_wxyz)


def inverse_align_quats(
    quats_wxyz: torch.Tensor,
    source: UpAxis,
) -> torch.Tensor:
    """Transform (N, 4) wxyz quaternions from internal back to *source*."""
    if source is UpAxis.Y_UP:
        return quats_wxyz
    from physics_sim.rotation_utils import rotmat_to_quat_wxyz, quat_mul_wxyz
    A_inv = inverse_alignment_matrix(source, quats_wxyz.device)
    q_inv = rotmat_to_quat_wxyz(A_inv.unsqueeze(0)).squeeze(0)
    return quat_mul_wxyz(q_inv.unsqueeze(0), quats_wxyz)


# ── Gravity ───────────────────────────────────────────────────────────

def gravity_vector(
    magnitude: float = 9.8,
    device: str = "cuda",
) -> torch.Tensor:
    """Gravity in internal (Y-up) coordinates: always ``[0, -mag, 0]``."""
    return torch.tensor([0.0, -magnitude, 0.0], device=device, dtype=torch.float32)


def gravity_from_source(
    g_source: list | tuple | torch.Tensor,
    source: UpAxis,
    device: str = "cuda",
) -> torch.Tensor:
    """Transform a gravity vector from *source* coordinates to internal Y-up."""
    g = torch.tensor(g_source, device=device, dtype=torch.float32) if not isinstance(g_source, torch.Tensor) else g_source.to(device)
    return align_directions(g.reshape(1, 3), source).squeeze(0)


# ── Camera extrinsic helpers (numpy) ──────────────────────────────────

def align_camera_w2c_rotation(
    R_w2c: np.ndarray,
    source: UpAxis,
) -> np.ndarray:
    """Adjust a W2C rotation matrix for the coordinate alignment.

    If world coords change by ``p_internal = A @ p_source``, then
    ``R_w2c_new = R_w2c @ A^T``.
    """
    A = alignment_matrix_np(source)
    return R_w2c @ A.T


def align_camera_position(
    position: np.ndarray,
    source: UpAxis,
) -> np.ndarray:
    """Transform a camera world-space position to internal coords."""
    A = alignment_matrix_np(source)
    return A @ position
