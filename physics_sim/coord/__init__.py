"""Coordinate API facade for source/internal frame conversion."""

from __future__ import annotations

from .axes import (
    PRESETS,
    UpAxis,
    SourceAxes,
    alignment_matrix,
    alignment_matrix_np,
    inverse_alignment_matrix,
    parse_axis,
)
from .gravity import (
    E_GRAVITY_AXIS,
    E_GRAVITY_MISSING,
    E_GRAVITY_SHAPE,
    gravity_contract_error,
    gravity_vector,
    normalize_internal_gravity,
)
from .ops import (
    align_camera_position,
    align_camera_w2c_rotation,
    align_covariances,
    align_directions,
    align_positions,
    align_quats,
    inverse_align_covariances,
    inverse_align_positions,
    inverse_align_quats,
)

__all__ = [
    "E_GRAVITY_AXIS",
    "E_GRAVITY_MISSING",
    "E_GRAVITY_SHAPE",
    "PRESETS",
    "SourceAxes",
    "UpAxis",
    "align_camera_position",
    "align_camera_w2c_rotation",
    "align_covariances",
    "align_directions",
    "align_positions",
    "align_quats",
    "alignment_matrix",
    "alignment_matrix_np",
    "gravity_contract_error",
    "gravity_vector",
    "inverse_align_covariances",
    "inverse_align_positions",
    "inverse_align_quats",
    "inverse_alignment_matrix",
    "normalize_internal_gravity",
    "parse_axis",
]
