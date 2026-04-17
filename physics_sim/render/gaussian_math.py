"""Gaussian math helpers shared by loader and colorizer."""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]
C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601,
      -0.6690465435572892, 0.10578554691520431, -0.6690465435572892,
      0.47308734787878004, -1.7701307697799304, 0.6258357354491761]


def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics at unit directions (up to degree 4)."""
    assert 0 <= deg <= 4
    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                      + C2[0] * xy * sh[..., 4]
                      + C2[1] * yz * sh[..., 5]
                      + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                      + C2[3] * xz * sh[..., 7]
                      + C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result
                          + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                          + C3[1] * xy * z * sh[..., 10]
                          + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                          + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                          + C3[5] * z * (xx - yy) * sh[..., 14]
                          + C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result
                              + C4[0] * xy * (xx - yy) * sh[..., 16]
                              + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                              + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                              + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                              + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                              + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                              + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                              + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                              + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Quaternion (N,4) -> rotation matrix (N,3,3)."""
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    rr, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.size(0), 3, 3), device="cuda")
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - rr * z)
    R[:, 0, 2] = 2 * (x * z + rr * y)
    R[:, 1, 0] = 2 * (x * y + rr * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - rr * x)
    R[:, 2, 0] = 2 * (x * z - rr * y)
    R[:, 2, 1] = 2 * (y * z + rr * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_covariance(
    scaling: torch.Tensor,
    rotation_quat: torch.Tensor,
    scaling_modifier: float = 1.0,
) -> torch.Tensor:
    """Compute upper-triangle covariance (N,6) from scaling + rotation."""
    L = torch.zeros((scaling.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(rotation_quat)
    L[:, 0, 0] = scaling_modifier * scaling[:, 0]
    L[:, 1, 1] = scaling_modifier * scaling[:, 1]
    L[:, 2, 2] = scaling_modifier * scaling[:, 2]
    L = R @ L
    cov = L @ L.transpose(1, 2)
    out = torch.zeros((cov.shape[0], 6), dtype=torch.float, device="cuda")
    out[:, 0] = cov[:, 0, 0]
    out[:, 1] = cov[:, 0, 1]
    out[:, 2] = cov[:, 0, 2]
    out[:, 3] = cov[:, 1, 1]
    out[:, 4] = cov[:, 1, 2]
    out[:, 5] = cov[:, 2, 2]
    return out
