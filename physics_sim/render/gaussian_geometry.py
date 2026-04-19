"""Geometry helpers for Gaussian covariance construction."""

from __future__ import annotations

import torch


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Quaternion `(N,4)` to rotation matrices `(N,3,3)`."""
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    rr, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    out = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - rr * z)
    out[:, 0, 2] = 2 * (x * z + rr * y)
    out[:, 1, 0] = 2 * (x * y + rr * z)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - rr * x)
    out[:, 2, 0] = 2 * (x * z - rr * y)
    out[:, 2, 1] = 2 * (y * z + rr * x)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def build_covariance(
    scaling: torch.Tensor,
    rotation_quat: torch.Tensor,
    scaling_modifier: float = 1.0,
) -> torch.Tensor:
    """Compute upper-triangle covariance `(N,6)` from scaling + rotation."""
    chol = torch.zeros((scaling.shape[0], 3, 3), dtype=scaling.dtype, device=scaling.device)
    chol[:, 0, 0] = scaling_modifier * scaling[:, 0]
    chol[:, 1, 1] = scaling_modifier * scaling[:, 1]
    chol[:, 2, 2] = scaling_modifier * scaling[:, 2]

    rot = build_rotation(rotation_quat)
    chol = rot @ chol
    cov = chol @ chol.transpose(1, 2)

    out = torch.zeros((cov.shape[0], 6), dtype=cov.dtype, device=cov.device)
    out[:, 0] = cov[:, 0, 0]
    out[:, 1] = cov[:, 0, 1]
    out[:, 2] = cov[:, 0, 2]
    out[:, 3] = cov[:, 1, 1]
    out[:, 4] = cov[:, 1, 2]
    out[:, 5] = cov[:, 2, 2]
    return out

