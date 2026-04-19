"""Boundary-condition normalization at stage boundary.

This module converts all user/collider boundary conditions into the
internal coordinate contract consumed by backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physics_sim.coord import SourceAxes, align_directions, align_positions
from physics_sim.geometry.plane_fit import fit_plane_svd
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _to_internal_surface_collider(bc: dict[str, Any], source_axes: SourceAxes) -> dict[str, Any]:
    """Convert one surface_collider BC to internal coordinates."""
    converted = dict(bc)
    point_src = torch.tensor(bc["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
    normal_src = torch.tensor(bc["normal"], device="cuda", dtype=torch.float32).reshape(1, 3)
    point_int = align_positions(point_src, source_axes)[0]
    normal_int = align_directions(normal_src, source_axes)[0]
    normal_int = normal_int / (torch.norm(normal_int) + 1e-12)
    converted["point"] = [float(x) for x in point_int.cpu().tolist()]
    converted["normal"] = [float(x) for x in normal_int.cpu().tolist()]
    converted.pop("space", None)
    return converted


def resolve_collider_boundary_conditions(collider_objects, source_axes: SourceAxes) -> list[dict[str, Any]]:
    """Convert collider_only objects into normalized internal BCs."""
    bc_list: list[dict[str, Any]] = []
    for obj in collider_objects:
        col = obj.collider or {}
        col_type = col.get("type", "plane")
        if col_type != "plane":
            raise ValueError(
                f"collider_only '{obj.name}' only supports type='plane' "
                f"(got {col_type!r})"
            )

        surface = col.get("surface", "sticky")
        friction = float(col.get("friction", 0.0))
        start_time = col.get("start_time", 0)
        end_time = col.get("end_time", 1e3)

        if col.get("point") is not None and col.get("normal") is not None:
            point = col["point"]
            normal = col["normal"]
            bc_list.append(
                _to_internal_surface_collider(
                    {
                        "type": "surface_collider",
                        "space": "world",
                        "point": point,
                        "normal": normal,
                        "surface": surface,
                        "friction": friction,
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                    source_axes=source_axes,
                )
            )
            continue

        pts = obj.positions.detach().cpu().numpy()
        if pts.shape[0] < 3:
            raise ValueError(
                f"collider_only '{obj.name}' has too few points for "
                "plane fitting and no explicit point+normal."
            )
        fit = col.get("fit") or {}
        sample_max = fit.get("sample_max", 200000)
        prefer_up_src = col.get("prefer_up", source_axes.up_vector.tolist())
        prefer_up_t = torch.tensor(prefer_up_src, dtype=torch.float32).reshape(1, 3)
        prefer_up_aligned = align_directions(prefer_up_t, source_axes)[0].numpy()
        res = fit_plane_svd(
            pts,
            sample_max=sample_max,
            seed=int(fit.get("seed", 0)),
            prefer_up=np.asarray(prefer_up_aligned, dtype=np.float32),
        )
        LOGGER.info(
            "[collider] name=%s plane_rms=%.6f point=%s normal=%s",
            obj.name,
            res.rms,
            res.point.tolist(),
            res.normal.tolist(),
        )
        bc_list.append(
            {
                "type": "surface_collider",
                "point": res.point.tolist(),
                "normal": res.normal.tolist(),
                "surface": surface,
                "friction": friction,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return bc_list


def normalize_boundary_conditions(
    *,
    raw_boundary_conditions: list[dict[str, Any]],
    collider_objects,
    source_axes: SourceAxes,
) -> list[dict[str, Any]]:
    """Normalize user BCs and collider BCs into one internal-only BC list."""
    merged = [dict(bc) for bc in raw_boundary_conditions]
    merged.extend(resolve_collider_boundary_conditions(collider_objects, source_axes))

    normalized: list[dict[str, Any]] = []
    for bc in merged:
        space = bc.get("space", "")
        if space == "world" and bc.get("type") == "surface_collider":
            normalized.append(_to_internal_surface_collider(bc, source_axes))
            continue
        if space in ("internal", ""):
            cleaned = dict(bc)
            cleaned.pop("space", None)
            normalized.append(cleaned)
            continue
        raise ValueError(
            f"unsupported boundary space={space!r} for type={bc.get('type')!r}; "
            "expected 'world' or 'internal'"
        )
    return normalized
