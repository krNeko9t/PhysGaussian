"""Shared boundary-plane helpers for Newton backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_surface_friction(
    *,
    surface: str = "slip",
    friction: Any | None = None,
) -> float:
    """Resolve collider friction from explicit value or surface preset."""
    if friction is not None:
        return float(friction)
    if surface == "sticky":
        return 1.0
    if surface == "slip":
        return 0.0
    return 0.5


def plane_from_point_normal(
    *,
    normal: list[float] | tuple[float, float, float],
    point: list[float] | tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Convert point-normal plane representation to Newton `ax+by+cz+d=0` tuple."""
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    d = -(nx * px + ny * py + nz * pz)
    return (nx, ny, nz, d)


def surface_plane_from_bc(
    bc: Mapping[str, Any],
) -> tuple[tuple[float, float, float, float], float]:
    """Build plane tuple and friction coefficient from a surface-collider BC."""
    plane = plane_from_point_normal(normal=bc["normal"], point=bc["point"])
    mu = resolve_surface_friction(
        surface=str(bc.get("surface", "slip")),
        friction=bc.get("friction"),
    )
    return plane, mu


def build_bounding_box_planes(
    *,
    lo: list[float] | tuple[float, float, float],
    hi: list[float] | tuple[float, float, float],
    margin: float,
) -> list[tuple[float, float, float, float]]:
    """Create six inward-facing planes that form an axis-aligned box."""
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    m = float(margin)
    return [
        (1.0, 0.0, 0.0, -(x0 - m)),
        (-1.0, 0.0, 0.0, x1 + m),
        (0.0, 1.0, 0.0, -(y0 - m)),
        (0.0, -1.0, 0.0, y1 + m),
        (0.0, 0.0, 1.0, -(z0 - m)),
        (0.0, 0.0, -1.0, z1 + m),
    ]

