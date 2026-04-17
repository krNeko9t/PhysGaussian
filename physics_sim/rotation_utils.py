"""
Unified rotation / quaternion utilities.

**Convention**: all quaternions in this repository use **scalar-first
(w, x, y, z)** order, matching the 3DGS PLY format.

This module consolidates quaternion math that was previously scattered
across ad-hoc helpers, ``renderer/gs_renderer.py``, and experiment
scripts.  Taichi kernels may keep their own ``@ti.func``
copies for GPU execution, but must mirror the logic here.
"""

from __future__ import annotations

import torch


# ── Quaternion arithmetic (wxyz) ──────────────────────────────────────

def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions.  Broadcasts ``a`` over ``b``."""
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


def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (= inverse for unit quaternions).  wxyz in/out."""
    signs = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    return q * signs


# ── Rotation-matrix <-> quaternion ────────────────────────────────────

def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix(ces) -> wxyz quaternion(s).

    Parameters
    ----------
    R : (3, 3) or (N, 3, 3)

    Returns
    -------
    (4,) or (N, 4)
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


def quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion(s) -> rotation matrix(ces).

    Parameters
    ----------
    q : (4,) or (N, 4)

    Returns
    -------
    (3, 3) or (N, 3, 3)
    """
    single = q.dim() == 1
    if single:
        q = q.unsqueeze(0)
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R[0] if single else R
