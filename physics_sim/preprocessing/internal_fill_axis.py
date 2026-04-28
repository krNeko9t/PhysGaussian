"""Signed axes for internal particle void-fill (voxel 6-neighbor topology).

Directions are interpreted in **internal simulation coordinates** (Y-up), same
as grid indices (x, y, z) in ``particle_filling``. Taichi kernels use
``dir_type`` in ``0..5``:

  0 ``+X``, 1 ``-X``, 2 ``+Y``, 3 ``-Y``, 4 ``+Z``, 5 ``-Z``

Semantic aliases assume right-handed Y-up with horizontal labels as **right /
left when facing -Z** (common graphics convention).
"""

from __future__ import annotations

from typing import Literal

_ALIAS_TO_CANONICAL: dict[str, str] = {
    "UP": "+Y",
    "DOWN": "-Y",
    "RIGHT": "+X",
    "LEFT": "-X",
}

_CANONICAL_AXES: tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

_AXIS_TO_DIR_INDEX: dict[str, int] = {a: i for i, a in enumerate(_CANONICAL_AXES)}

InternalFillSignedAxis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def normalize_fill_axis(s: str) -> str:
    """Return canonical ``+X``/``-Z`` string; accept aliases ``up``/``down``/etc."""
    key = s.strip().upper()
    if key in _ALIAS_TO_CANONICAL:
        key = _ALIAS_TO_CANONICAL[key]
    if key not in _AXIS_TO_DIR_INDEX:
        allowed = list(_CANONICAL_AXES) + list(_ALIAS_TO_CANONICAL.keys())
        raise ValueError(
            f"Unknown internal fill axis {s!r}; expected one of {allowed}"
        )
    return key


def normalize_fill_axis_optional(v: str | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return normalize_fill_axis(v)


def void_probe_skip_axis_to_dir_index(axis: str | None) -> int:
    """Neighbor direction excluded from shell probe; ``-1`` = exclude none."""
    if axis is None:
        return -1
    return _AXIS_TO_DIR_INDEX[normalize_fill_axis(axis)]


def void_fill_ray_axis_to_dir_index(axis: str) -> int:
    return _AXIS_TO_DIR_INDEX[normalize_fill_axis(axis)]
