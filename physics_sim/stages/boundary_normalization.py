"""Boundary-condition normalization at stage boundary.

This module converts all user/collider boundary conditions into the
internal coordinate contract consumed by backends. Inputs and outputs
are typed Pydantic ``BoundaryCondition`` instances; after normalization
``SurfaceCollider.space`` is ``"internal"`` and ``point``/``normal`` are
expressed in the internal Y-up frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physics_sim.config.models import (
    BoundaryCondition,
    SurfaceCollider,
)
from physics_sim.coord import SourceAxes, align_directions, align_positions
from physics_sim.geometry.plane_fit import fit_plane_svd
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _transform_surface_collider(
    bc: SurfaceCollider, source_axes: SourceAxes,
) -> SurfaceCollider:
    """Transform a world-space surface_collider to internal coordinates."""
    point_src = torch.tensor(bc.point, device="cuda", dtype=torch.float32).reshape(1, 3)
    normal_src = torch.tensor(bc.normal, device="cuda", dtype=torch.float32).reshape(1, 3)
    point_int = align_positions(point_src, source_axes)[0]
    normal_int = align_directions(normal_src, source_axes)[0]
    normal_int = normal_int / (torch.norm(normal_int) + 1e-12)
    return bc.model_copy(update={
        "point": tuple(float(x) for x in point_int.cpu().tolist()),
        "normal": tuple(float(x) for x in normal_int.cpu().tolist()),
        "space": "internal",
    })


def resolve_collider_boundary_conditions(
    collider_objects, source_axes: SourceAxes,
) -> list[BoundaryCondition]:
    """Convert collider_only objects into normalized internal BCs."""
    bc_list: list[BoundaryCondition] = []
    for obj in collider_objects:
        col = obj.collider
        if col is None:
            raise ValueError(
                f"collider_only '{obj.name}' has no collider config"
            )
        if col.type != "plane":
            raise ValueError(
                f"collider_only '{obj.name}' only supports type='plane' "
                f"(got {col.type!r})"
            )

        if col.point is not None and col.normal is not None:
            world_bc = SurfaceCollider(
                point=col.point,
                normal=col.normal,
                surface=col.surface,
                friction=float(col.friction),
                space="world",
                start_time=col.start_time,
                end_time=col.end_time,
            )
            bc_list.append(_transform_surface_collider(world_bc, source_axes))
            continue

        pts = obj.positions.detach().cpu().numpy()
        if pts.shape[0] < 3:
            raise ValueError(
                f"collider_only '{obj.name}' has too few points for "
                "plane fitting and no explicit point+normal."
            )
        fit = col.fit
        sample_max = fit.sample_max if fit is not None else 200_000
        seed = fit.seed if fit is not None else 0
        prefer_up_src = (
            list(col.prefer_up) if col.prefer_up is not None
            else source_axes.up_vector.tolist()
        )
        prefer_up_t = torch.tensor(prefer_up_src, dtype=torch.float32).reshape(1, 3)
        prefer_up_aligned = align_directions(prefer_up_t, source_axes)[0].numpy()
        res = fit_plane_svd(
            pts,
            sample_max=sample_max,
            seed=int(seed),
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
            SurfaceCollider(
                point=tuple(float(x) for x in res.point.tolist()),
                normal=tuple(float(x) for x in res.normal.tolist()),
                surface=col.surface,
                friction=float(col.friction),
                space="internal",
                start_time=col.start_time,
                end_time=col.end_time,
            )
        )
    return bc_list


def normalize_boundary_conditions(
    *,
    raw_boundary_conditions: list[BoundaryCondition],
    collider_objects,
    source_axes: SourceAxes,
) -> list[BoundaryCondition]:
    """Normalize user BCs and collider BCs into one internal-only BC list.

    After this call:
    - ``SurfaceCollider`` entries have ``space="internal"`` and their
      ``point``/``normal`` expressed in the internal Y-up frame.
    - ``BoundingBox`` / ``ReleaseParticlesSequentially`` pass through
      unchanged (no spatial coordinates on BoundingBox; release params
      are already in the internal convention).
    """
    normalized: list[BoundaryCondition] = []
    for bc in raw_boundary_conditions:
        if isinstance(bc, SurfaceCollider):
            if bc.space == "world":
                normalized.append(_transform_surface_collider(bc, source_axes))
            else:
                normalized.append(bc)
        else:
            normalized.append(bc)

    normalized.extend(resolve_collider_boundary_conditions(collider_objects, source_axes))
    return normalized
