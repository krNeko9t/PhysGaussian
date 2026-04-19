"""Shared spherical-harmonics tensor/layout contract helpers."""

from __future__ import annotations

import numpy as np
import torch

_MAX_SH_DEGREE = 4


def validate_sh_degree(sh_degree: int) -> None:
    """Validate SH degree supported by the current evaluator implementation."""
    if sh_degree < 0 or sh_degree > _MAX_SH_DEGREE:
        raise ValueError(
            f"Unsupported SH degree {sh_degree}. "
            f"Current implementation supports [0, {_MAX_SH_DEGREE}]."
        )


def sh_coeff_count(sh_degree: int) -> int:
    """Return the number of SH coefficients per color channel."""
    validate_sh_degree(sh_degree)
    return (sh_degree + 1) ** 2


def expected_f_rest_feature_count(sh_degree: int) -> int:
    """Return expected flattened PLY `f_rest_*` feature count."""
    return 3 * sh_coeff_count(sh_degree) - 3


def assemble_sh_from_ply_features(
    features_dc: np.ndarray,
    features_rest: np.ndarray,
    sh_degree: int,
) -> np.ndarray:
    """Build SH tensor with canonical shape `(N, C, 3)` from PLY arrays."""
    coeffs = sh_coeff_count(sh_degree)
    expected_rest = expected_f_rest_feature_count(sh_degree)
    if features_dc.ndim != 3 or features_dc.shape[1:] != (3, 1):
        raise ValueError(
            f"features_dc must have shape (N, 3, 1), got {features_dc.shape}."
        )
    if features_rest.ndim != 2 or features_rest.shape[1] != expected_rest:
        raise ValueError(
            "features_rest must have shape "
            f"(N, {expected_rest}), got {features_rest.shape}."
        )
    if features_rest.shape[0] != features_dc.shape[0]:
        raise ValueError(
            "features_dc and features_rest must share the same leading dimension."
        )
    features_rest_reshaped = features_rest.reshape(features_rest.shape[0], 3, coeffs - 1)
    shs = np.concatenate([features_dc, features_rest_reshaped], axis=2)
    return np.transpose(shs, (0, 2, 1))


def flatten_sh_coeffs(shs: torch.Tensor) -> torch.Tensor:
    """Flatten SH tensor `(N, C, 3)` to `(N, C*3)`."""
    if shs.ndim != 3 or shs.shape[-1] != 3:
        raise ValueError(f"shs must have shape (N, C, 3), got {tuple(shs.shape)}.")
    return shs.reshape(shs.shape[0], -1)


def restore_sh_coeffs(flat_shs: torch.Tensor) -> torch.Tensor:
    """Restore flattened SH tensor `(N, C*3)` back to `(N, C, 3)`."""
    if flat_shs.ndim != 2:
        raise ValueError(f"flat_shs must have shape (N, C*3), got {tuple(flat_shs.shape)}.")
    if flat_shs.shape[1] % 3 != 0:
        raise ValueError(
            f"flat_shs second dimension must be divisible by 3, got {flat_shs.shape[1]}."
        )
    return flat_shs.view(flat_shs.shape[0], -1, 3)

