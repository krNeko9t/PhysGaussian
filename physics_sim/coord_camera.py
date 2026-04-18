"""Camera-specific coordinate helpers."""

from __future__ import annotations

import numpy as np


def align_camera_w2c_rotation(
    R_w2c: np.ndarray,
    alignment_matrix_np: np.ndarray,
) -> np.ndarray:
    """Adjust W2C rotation matrix after world-coordinate alignment."""
    return R_w2c @ alignment_matrix_np.T


def align_camera_position(
    position: np.ndarray,
    alignment_matrix_np: np.ndarray,
) -> np.ndarray:
    """Transform camera world position with alignment matrix."""
    return alignment_matrix_np @ position
