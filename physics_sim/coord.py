"""
Coordinate system conventions and conversions.

This module is the **single source of truth** for all coordinate-system
transformations in this repository.

Internal convention (used everywhere after ingestion)
-----------------------------------------------------
Right-handed, **Y-up**::

    +X = right
    +Y = up
    +Z = toward viewer  (OpenGL style)

Source convention
-----------------
Described by **two axes**: ``source_up`` and ``source_front``.

*  ``source_up``   – which axis in the source data points *upward*
   (determines gravity direction).
*  ``source_front`` – which axis in the source data points *toward the
   viewer* (determines the scene's facing direction and orbit-camera
   azimuth-zero).

The third axis (*right*) is derived via ``right = cross(up, front)``
to guarantee a right-handed system.

Common presets::

    OPENGL        up="+Y"  front="+Z"     (identity – same as internal)
    BLENDER       up="+Z"  front="-Y"     (Blender default world)
    Z_UP_X_FRONT  up="+Z"  front="+X"

All functions operate on **torch** tensors to match the rest of the
pipeline.  Lightweight numpy helpers are provided for camera math.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from physics_sim.coord_camera import (
    align_camera_position as _align_camera_position_impl,
    align_camera_w2c_rotation as _align_camera_w2c_rotation_impl,
)
from physics_sim.coord_gravity import (
    E_GRAVITY_AXIS,
    E_GRAVITY_MISSING,
    E_GRAVITY_SHAPE,
    gravity_contract_error,
    gravity_vector,
    normalize_internal_gravity,
)


# ── Axis parsing ─────────────────────────────────────────────────────

_AXIS_MAP = {
    "+X": [1, 0, 0], "-X": [-1, 0, 0],
    "+Y": [0, 1, 0], "-Y": [0, -1, 0],
    "+Z": [0, 0, 1], "-Z": [0, 0, -1],
}


def parse_axis(s: str) -> np.ndarray:
    """Parse ``"+X"``, ``"-Z"`` etc. into a unit vector."""
    key = s.strip().upper()
    if key in _AXIS_MAP:
        return np.array(_AXIS_MAP[key], dtype=np.float32)
    raise ValueError(
        f"Unknown axis string: '{s}'.  "
        f"Expected one of {list(_AXIS_MAP.keys())}"
    )


def _normalise_up_string(s: str) -> str:
    """Accept legacy ``'Y_UP'``, ``'Z_UP'``, ``'+Y'``, ``'Y'`` etc."""
    s = s.strip().upper().replace("_UP", "").replace("UP", "")
    s = s.replace("_", "").replace(" ", "")
    if s in ("Y", "+Y"):
        return "+Y"
    if s in ("Z", "+Z"):
        return "+Z"
    if s in ("X", "+X"):
        return "+X"
    if s.startswith(("+", "-")) and len(s) == 2 and s[1] in "XYZ":
        return s
    raise ValueError(f"Cannot interpret up-axis string: '{s}'")


# ── Legacy UpAxis enum (kept for backward compatibility) ─────────────

class UpAxis(enum.Enum):
    Y_UP = "Y_UP"
    Z_UP = "Z_UP"


# ── Named presets ────────────────────────────────────────────────────

PRESETS: dict[str, dict[str, str]] = {
    "OPENGL":       dict(up="+Y", front="+Z"),
    "BLENDER":      dict(up="+Z", front="-Y"),
    "Z_UP_X_FRONT": dict(up="+Z", front="+X"),
    "Z_UP_Y_FRONT": dict(up="+Z", front="+Y"),
}


# ── SourceAxes: the core abstraction ─────────────────────────────────

@dataclass(frozen=True)
class SourceAxes:
    """Fully describes how source data maps to the internal Y-up frame.

    Construct via :meth:`from_config` (the recommended entry-point).

    Attributes
    ----------
    up_str, front_str : str
        Canonical axis strings, e.g. ``"+Z"``, ``"-Y"``.
    A : torch.Tensor
        3x3 alignment matrix.  ``pos_internal = pos_source @ A.T``
        (row-vector convention).
    A_inv : torch.Tensor
        Inverse of ``A`` (= ``A.T`` for orthogonal matrices).
    is_identity : bool
        True when source == internal (no transform needed).
    """
    up_str: str
    front_str: str
    A: torch.Tensor          # (3, 3) float32, CPU
    A_inv: torch.Tensor      # (3, 3) float32, CPU
    is_identity: bool

    # ── Factories ────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        source_up: str = "+Y",
        source_front: str | None = None,
    ) -> SourceAxes:
        """Build from user config strings.

        Parameters
        ----------
        source_up : str
            Which source axis is "up".  Accepts ``"+Z"``, ``"Z_UP"``,
            ``"Z"``, etc.
        source_front : str or None
            Which source axis faces the viewer.  If *None*, a sensible
            default is chosen based on ``source_up`` to match the legacy
            ``_z_up_to_y_up_matrix`` behaviour.
        """
        up_s = _normalise_up_string(source_up)

        if source_front is None:
            source_front = _default_front(up_s)
        front_s = source_front.strip().upper()
        if front_s not in _AXIS_MAP:
            raise ValueError(f"Invalid source_front: '{source_front}'")

        return cls._build(up_s, front_s)

    @classmethod
    def from_preset(cls, name: str) -> SourceAxes:
        """Build from a named preset (e.g. ``"BLENDER"``)."""
        name = name.strip().upper()
        if name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'.  "
                f"Available: {list(PRESETS.keys())}"
            )
        p = PRESETS[name]
        return cls._build(p["up"], p["front"])

    @classmethod
    def identity(cls) -> SourceAxes:
        """Source == internal (no transform)."""
        I = torch.eye(3, dtype=torch.float32)
        return cls(
            up_str="+Y", front_str="+Z",
            A=I, A_inv=I, is_identity=True,
        )

    # ── Internal builder ─────────────────────────────────────────

    @classmethod
    def _build(cls, up_s: str, front_s: str) -> SourceAxes:
        up_vec = parse_axis(up_s)
        front_vec = parse_axis(front_s)

        if abs(np.dot(up_vec, front_vec)) > 1e-6:
            raise ValueError(
                f"source_up='{up_s}' and source_front='{front_s}' "
                f"are not orthogonal."
            )

        right_vec = np.cross(up_vec, front_vec)
        norm = np.linalg.norm(right_vec)
        if norm < 1e-6:
            raise ValueError(
                f"Degenerate axes: up={up_s}, front={front_s}"
            )
        right_vec = right_vec / norm

        B_src = np.column_stack([right_vec, up_vec, front_vec])
        A_np = B_src.T.astype(np.float32)  # B_src is orthonormal → A = B^T
        A = torch.from_numpy(A_np)
        A_inv = A.T.contiguous()

        is_id = bool(torch.allclose(A, torch.eye(3), atol=1e-6))

        return cls(
            up_str=up_s, front_str=front_s,
            A=A, A_inv=A_inv, is_identity=is_id,
        )

    # ── Convenience ──────────────────────────────────────────────

    def to(self, device: str | torch.device) -> SourceAxes:
        """Return a copy with tensors on *device*."""
        return SourceAxes(
            up_str=self.up_str,
            front_str=self.front_str,
            A=self.A.to(device),
            A_inv=self.A_inv.to(device),
            is_identity=self.is_identity,
        )

    @property
    def up_vector(self) -> np.ndarray:
        """Source up-axis as a unit vector in source coordinates."""
        return parse_axis(self.up_str)


def _default_front(up_s: str) -> str:
    """Choose a default front axis given the up axis.

    For ``+Z`` up, front defaults to ``-Y`` (matching the legacy
    ``_z_up_to_y_up_matrix`` and Blender convention).
    """
    defaults = {
        "+Y": "+Z",
        "-Y": "-Z",
        "+Z": "-Y",
        "-Z": "+Y",
        "+X": "+Z",
        "-X": "-Z",
    }
    if up_s in defaults:
        return defaults[up_s]
    raise ValueError(f"No default front for up='{up_s}'; specify source_front explicitly.")


# ── Alignment matrix accessors ───────────────────────────────────────

def alignment_matrix(axes: SourceAxes, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return 3x3 ``A`` such that ``pos_internal = pos_source @ A.T``."""
    return axes.A.to(device)


def inverse_alignment_matrix(axes: SourceAxes, device: str | torch.device = "cpu") -> torch.Tensor:
    """``A_inv`` such that ``pos_source = pos_internal @ A_inv.T``."""
    return axes.A_inv.to(device)


def alignment_matrix_np(axes: SourceAxes) -> np.ndarray:
    """Numpy variant of :func:`alignment_matrix` (for camera math)."""
    return axes.A.numpy()


# ── Batch position / direction helpers ────────────────────────────────

def align_positions(
    positions: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform (N, 3) positions from *source* to internal Y-up."""
    if axes.is_identity:
        return positions
    A = axes.A.to(positions.device)
    return positions @ A.T


def inverse_align_positions(
    positions: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform (N, 3) positions from internal Y-up back to *source*."""
    if axes.is_identity:
        return positions
    A_inv = axes.A_inv.to(positions.device)
    return positions @ A_inv.T


def align_directions(
    dirs: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform direction vectors (normals, gravity, …) from source to internal."""
    return align_positions(dirs, axes)


# ── Covariance alignment ─────────────────────────────────────────────

def align_covariances(
    cov_upper: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform (N, 6) upper-triangle covariances from *source* to internal.

    Upper-triangle layout: ``[c00, c01, c02, c11, c12, c22]``.
    Cov transforms as ``C' = A @ C @ A^T``.
    """
    if axes.is_identity:
        return cov_upper
    A = axes.A.to(cov_upper.device)
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
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform (N, 6) covariances from internal Y-up back to *source*."""
    if axes.is_identity:
        return cov_upper
    A_inv = axes.A_inv.to(cov_upper.device)
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
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform (N, 4) wxyz quaternions from *source* to internal Y-up.

    ``q_internal = q_align * q_source`` (Hamilton product).
    """
    if axes.is_identity:
        return quats_wxyz
    from physics_sim.rotation_utils import rotmat_to_quat_wxyz, quat_mul_wxyz
    A = axes.A.to(quats_wxyz.device)
    q_align = rotmat_to_quat_wxyz(A.unsqueeze(0)).squeeze(0)
    return quat_mul_wxyz(q_align.unsqueeze(0), quats_wxyz)


def inverse_align_quats(
    quats_wxyz: torch.Tensor,
    axes: SourceAxes,
) -> torch.Tensor:
    """Transform (N, 4) wxyz quaternions from internal back to *source*."""
    if axes.is_identity:
        return quats_wxyz
    from physics_sim.rotation_utils import rotmat_to_quat_wxyz, quat_mul_wxyz
    A_inv = axes.A_inv.to(quats_wxyz.device)
    q_inv = rotmat_to_quat_wxyz(A_inv.unsqueeze(0)).squeeze(0)
    return quat_mul_wxyz(q_inv.unsqueeze(0), quats_wxyz)


# ── Camera extrinsic helpers (numpy) ──────────────────────────────────

def align_camera_w2c_rotation(
    R_w2c: np.ndarray,
    axes: SourceAxes,
) -> np.ndarray:
    """Adjust a W2C rotation matrix for the coordinate alignment.

    If world coords change by ``p_internal = A @ p_source``, then
    ``R_w2c_new = R_w2c @ A^T``.
    """
    A = alignment_matrix_np(axes)
    return _align_camera_w2c_rotation_impl(R_w2c, A)


def align_camera_position(
    position: np.ndarray,
    axes: SourceAxes,
) -> np.ndarray:
    """Transform a camera world-space position to internal coords."""
    A = alignment_matrix_np(axes)
    return _align_camera_position_impl(position, A)
