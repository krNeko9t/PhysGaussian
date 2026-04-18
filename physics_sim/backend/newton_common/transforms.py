"""Shared Newton backend transform helpers.

This module centralizes quaternion and covariance conversions that were
previously duplicated in multiple backend solver files.
"""

from __future__ import annotations

import numpy as np
import torch


def rotmat_to_wp_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to Warp quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = torch.zeros((3, 3), device=q.device, dtype=q.dtype)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compose_body_quat_wxyz(
    body_quat_xyzw: torch.Tensor,
    init_quats_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Compose body quaternion with per-particle quaternions in wxyz."""
    bx, by, bz, bw = (
        body_quat_xyzw[0],
        body_quat_xyzw[1],
        body_quat_xyzw[2],
        body_quat_xyzw[3],
    )
    iw, ix, iy, iz = (
        init_quats_wxyz[:, 0],
        init_quats_wxyz[:, 1],
        init_quats_wxyz[:, 2],
        init_quats_wxyz[:, 3],
    )
    ow = bw * iw - bx * ix - by * iy - bz * iz
    ox = bw * ix + bx * iw + by * iz - bz * iy
    oy = bw * iy - bx * iz + by * iw + bz * ix
    oz = bw * iz + bx * iy - by * ix + bz * iw
    out = torch.stack([ow, ox, oy, oz], dim=1)
    return torch.nn.functional.normalize(out, dim=1)


def unpack_cov6_to_3x3(cov6: torch.Tensor) -> torch.Tensor:
    """Convert (N, 6) upper-triangle covariance to (N, 3, 3)."""
    N = cov6.shape[0]
    out = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    out[:, 0, 0] = cov6[:, 0]
    out[:, 0, 1] = cov6[:, 1]
    out[:, 0, 2] = cov6[:, 2]
    out[:, 1, 0] = cov6[:, 1]
    out[:, 1, 1] = cov6[:, 3]
    out[:, 1, 2] = cov6[:, 4]
    out[:, 2, 0] = cov6[:, 2]
    out[:, 2, 1] = cov6[:, 4]
    out[:, 2, 2] = cov6[:, 5]
    return out


def pack_cov3x3_to_6(cov3x3: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) covariance to (N, 6) upper-triangle format."""
    out = torch.zeros((cov3x3.shape[0], 6), device=cov3x3.device, dtype=cov3x3.dtype)
    out[:, 0] = cov3x3[:, 0, 0]
    out[:, 1] = cov3x3[:, 0, 1]
    out[:, 2] = cov3x3[:, 0, 2]
    out[:, 3] = cov3x3[:, 1, 1]
    out[:, 4] = cov3x3[:, 1, 2]
    out[:, 5] = cov3x3[:, 2, 2]
    return out
