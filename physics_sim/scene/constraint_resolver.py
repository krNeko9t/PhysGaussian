"""Resolve scene-graph ``Constraint`` declarations to concrete runtime data.

This module bridges the config layer (``physics_sim.config.scene``) and
the backend layer (``physics_sim.backend.*``):

Config layer declares, e.g.::

    PinToWorld(particles=ProximitySelector(
        part='branches', to_surface_of='pot', max_distance=0.02))

After scene assembly we know the concrete per-part positions and global
index ranges, so a *resolved* constraint looks like::

    ResolvedPinToWorld(particle_indices=np.array([432, 433, ...]))

Each backend's ``apply_constraints`` method consumes the resolved list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from scipy.spatial import cKDTree

from physics_sim.config.scene import (
    BoxSelector,
    CollideOnly,
    IndexSelector,
    PinToBody,
    PinToWorld,
    ProximitySelector,
    SceneConfig,
)
from physics_sim.backend.spec import PartRuntimeInfo
from physics_sim.scene.objects import SceneObject


# ── Resolved constraint dataclasses ────────────────────────────────

@dataclass
class ResolvedPinToWorld:
    """Particles (identified by global model index) to freeze at spawn pose."""
    particle_indices: np.ndarray  # dtype=int64, 1-D
    part_name: str                # source part (for diagnostics)


@dataclass
class ResolvedPinToBody:
    """Particles to freeze in a rigid body's local frame.

    Carries the world positions at resolution time; each backend converts
    them to body-local offsets once it knows the target body's ``body_q``.
    """
    particle_indices: np.ndarray            # (N,) int64, global indices
    particle_world_positions: np.ndarray    # (N, 3) float64
    part_name: str                          # source part (for diagnostics)
    body_part_name: str                     # rigid part providing the body


@dataclass
class ResolvedCollideOnly:
    """Part name that must be treated as collide-only in the backend."""
    part_name: str


ResolvedConstraint = Union[
    ResolvedPinToWorld, ResolvedPinToBody, ResolvedCollideOnly,
]


# ── Part lookup helpers ────────────────────────────────────────────

@dataclass
class _PartLookup:
    """Everything the resolver needs to translate a selector into global indices."""
    name: str
    positions: np.ndarray   # (N, 3) float64, world (internal-Y-up) coords
    global_offset: int | None  # start index in the final model; None if not a sim part


def _build_part_lookup(
    scene_objects: Sequence[SceneObject],
    parts_runtime: Sequence[PartRuntimeInfo],
) -> dict[str, _PartLookup]:
    """Index scene objects by name, reading positions to CPU once."""
    offset_by_name = {info.name: info.particle_indices[0] for info in parts_runtime
                      if info.particle_indices}
    out: dict[str, _PartLookup] = {}
    for obj in scene_objects:
        pos_np = obj.positions.detach().cpu().numpy().astype(np.float64, copy=False)
        out[obj.name] = _PartLookup(
            name=obj.name,
            positions=pos_np,
            global_offset=offset_by_name.get(obj.name),
        )
    return out


# ── Selector resolution ────────────────────────────────────────────

def _resolve_selector(
    selector,
    parts: dict[str, _PartLookup],
) -> np.ndarray:
    """Return a 1-D int64 array of **local** particle indices in selector.part."""
    if isinstance(selector, ProximitySelector):
        src = parts[selector.part].positions
        ref = parts[selector.to_surface_of].positions
        if src.size == 0 or ref.size == 0:
            return np.zeros(0, dtype=np.int64)
        # KD-tree: for each src point, find its nearest ref neighbour in
        # O(log M) rather than O(M).  Points with no neighbour within
        # ``max_distance`` come back with distance = +inf.
        tree = cKDTree(ref)
        dists, _ = tree.query(
            src, k=1, distance_upper_bound=float(selector.max_distance),
        )
        return np.nonzero(np.isfinite(dists))[0].astype(np.int64)

    if isinstance(selector, BoxSelector):
        if selector.space != "world":
            raise NotImplementedError(
                "BoxSelector with space='part_local' is not implemented yet "
                "(requires access to the part's original untransformed positions)"
            )
        src = parts[selector.part].positions
        lo = np.asarray(selector.min, dtype=np.float64)
        hi = np.asarray(selector.max, dtype=np.float64)
        mask = np.all((src >= lo) & (src <= hi), axis=1)
        return np.nonzero(mask)[0].astype(np.int64)

    if isinstance(selector, IndexSelector):
        if isinstance(selector.indices, str):
            path = Path(selector.indices)
            if not path.exists():
                raise FileNotFoundError(
                    f"IndexSelector: indices file not found: {selector.indices}"
                )
            arr = np.load(path).astype(np.int64)
        else:
            arr = np.asarray(selector.indices, dtype=np.int64)
        n_local = parts[selector.part].positions.shape[0]
        bad = (arr < 0) | (arr >= n_local)
        if bad.any():
            raise ValueError(
                f"IndexSelector for part '{selector.part}': indices out of range "
                f"[0, {n_local}); {int(bad.sum())} bad entries"
            )
        return arr

    raise TypeError(f"Unknown selector type: {type(selector).__name__}")


def _resolve_to_global(
    local_indices: np.ndarray,
    part: _PartLookup,
    *,
    ctx: str,
) -> np.ndarray:
    """Offset local particle indices into the concatenated global model."""
    if part.global_offset is None:
        raise ValueError(
            f"{ctx}: part '{part.name}' is not a simulated part (no particles "
            "in the final model); it cannot be the source of a particle selector"
        )
    return (local_indices + part.global_offset).astype(np.int64)


# ── Public API ─────────────────────────────────────────────────────

def resolve_constraints(
    scene: SceneConfig,
    scene_objects: Sequence[SceneObject],
    parts_runtime: Sequence[PartRuntimeInfo],
) -> list[ResolvedConstraint]:
    """Translate every ``ConstraintConfig`` in *scene* to a ``ResolvedConstraint``.

    Parameters
    ----------
    scene : SceneConfig
        The scene-graph declaration.
    scene_objects : list[SceneObject]
        All assembled parts (sim + static + collider), positions in internal
        Y-up coords.  Used to look up positions for ProximitySelector.
    parts_runtime : list[PartRuntimeInfo]
        Per-sim-part runtime descriptors (provides ``particle_indices``
        = global offset in the concatenated model).
    """
    parts = _build_part_lookup(scene_objects, parts_runtime)

    out: list[ResolvedConstraint] = []
    for i, c in enumerate(scene.constraints):
        ctx = f"constraint[{i}] {type(c).__name__}"

        if isinstance(c, CollideOnly):
            if c.part not in parts:
                raise ValueError(f"{ctx}: unknown part '{c.part}'")
            out.append(ResolvedCollideOnly(part_name=c.part))
            continue

        if isinstance(c, (PinToWorld, PinToBody)):
            sel_part_name = c.particles.part
            if sel_part_name not in parts:
                raise ValueError(f"{ctx}: unknown selector part '{sel_part_name}'")
            local_idx = _resolve_selector(c.particles, parts)
            if local_idx.size == 0:
                raise ValueError(
                    f"{ctx}: selector matched zero particles; check the selector "
                    "spec (too strict threshold, empty box, or IndexSelector with "
                    "unused range)"
                )
            sel_part = parts[sel_part_name]
            global_idx = _resolve_to_global(local_idx, sel_part, ctx=ctx)

            if isinstance(c, PinToWorld):
                out.append(ResolvedPinToWorld(
                    particle_indices=global_idx,
                    part_name=sel_part_name,
                ))
            else:  # PinToBody
                if c.body not in parts:
                    raise ValueError(f"{ctx}: unknown body part '{c.body}'")
                world_pos = sel_part.positions[local_idx].astype(np.float64, copy=False)
                out.append(ResolvedPinToBody(
                    particle_indices=global_idx,
                    particle_world_positions=world_pos,
                    part_name=sel_part_name,
                    body_part_name=c.body,
                ))
            continue

        raise TypeError(f"{ctx}: unsupported constraint type {type(c).__name__}")

    return out
