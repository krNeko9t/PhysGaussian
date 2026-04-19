"""Spherical harmonics evaluation helpers."""

from __future__ import annotations

import torch

from physics_sim.sh_contract import validate_sh_degree

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def _eval_degree_1_terms(
    sh: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    return -C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]


def _eval_degree_2_terms(
    sh: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    xy: torch.Tensor,
    yz: torch.Tensor,
    xz: torch.Tensor,
) -> torch.Tensor:
    return (
        C2[0] * xy * sh[..., 4]
        + C2[1] * yz * sh[..., 5]
        + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
        + C2[3] * xz * sh[..., 7]
        + C2[4] * (xx - yy) * sh[..., 8]
    )


def _eval_degree_3_terms(
    sh: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    return (
        C3[0] * y * (3 * xx - yy) * sh[..., 9]
        + C3[1] * xy * z * sh[..., 10]
        + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
        + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
        + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
        + C3[5] * z * (xx - yy) * sh[..., 14]
        + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
    )


def _eval_degree_4_terms(
    sh: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    xy: torch.Tensor,
    yz: torch.Tensor,
    xz: torch.Tensor,
) -> torch.Tensor:
    return (
        C4[0] * xy * (xx - yy) * sh[..., 16]
        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
        + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24]
    )


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH at unit directions, up to degree 4."""
    validate_sh_degree(deg)
    result = C0 * sh[..., 0]
    if deg == 0:
        return result

    x = dirs[..., 0:1]
    y = dirs[..., 1:2]
    z = dirs[..., 2:3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z

    result = result + _eval_degree_1_terms(sh, x, y, z)
    if deg >= 2:
        result = result + _eval_degree_2_terms(sh, xx, yy, zz, xy, yz, xz)
    if deg >= 3:
        result = result + _eval_degree_3_terms(sh, x, y, z, xx, yy, zz, xy)
    if deg >= 4:
        result = result + _eval_degree_4_terms(sh, x, y, z, xx, yy, zz, xy, yz, xz)
    return result

