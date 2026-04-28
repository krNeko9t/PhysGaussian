"""Particle-filling orchestration for the scene-setup stage.

Adds synthetic interior particles to :class:`SceneObject`\\ s whose
``PartConfig`` declared a ``particle_filling`` config.  Parts sharing
the same ``fill_group`` are unioned before filling so the interior of
a logically-single body isn't torn by seams; filled particles are then
reassigned to their owning part via nearest-neighbor.

Attribute inheritance (SH / opacity / covariance / quat / scale) is
pluggable via :class:`FillStrategy`.  The default is nearest-neighbor.

This module only consumes scene-layer objects and the preprocessing
filling utility — it must not import backend or render internals.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import torch

from physics_sim.config.models import FillingConfig
from physics_sim.errors import configuration_error
from physics_sim.scene.objects import SceneObject


# ── Nearest-neighbor helper (pure torch, chunked) ───────────────────────

def _nearest_src_index(
    src_pos: torch.Tensor,
    new_pos: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """Return (N_new,) long tensor: index into src_pos of each new_pos's nearest.

    Chunked to cap the (chunk, N_src) distance matrix memory.
    """
    out = torch.empty(new_pos.shape[0], dtype=torch.long, device=new_pos.device)
    for start in range(0, new_pos.shape[0], chunk):
        end = min(start + chunk, new_pos.shape[0])
        d = torch.cdist(new_pos[start:end], src_pos)
        out[start:end] = d.argmin(dim=1)
    return out


# ── Strategy protocol + default implementation ──────────────────────────

class FillStrategy(Protocol):
    """Compute per-new-particle attributes from a group's source particles.

    Implementations receive the group-wide concatenated source tensors and
    must return a dict mirroring ``src_attrs`` with leading axis of size
    ``new_pos.shape[0]``.
    """

    def inherit(
        self,
        *,
        src_pos: torch.Tensor,
        src_attrs: dict[str, torch.Tensor],
        new_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...


class NearestNeighborStrategy:
    """Copy every src attribute from each new particle's nearest source.

    Attribute-agnostic: whatever keys the caller puts in ``src_attrs`` come
    back gathered by the NN indices.  The special key ``part_id`` drives
    post-filling partitioning back to the owning part.
    """

    def inherit(
        self,
        *,
        src_pos: torch.Tensor,
        src_attrs: dict[str, torch.Tensor],
        new_pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        idx = _nearest_src_index(src_pos, new_pos)
        return {k: v[idx] for k, v in src_attrs.items()}


# ── Boundary / grouping ─────────────────────────────────────────────────

def _derive_boundary(pos: torch.Tensor, margin_frac: float = 0.05) -> list[float]:
    lo = pos.min(dim=0).values
    hi = pos.max(dim=0).values
    pad = (hi - lo).max() * margin_frac
    lo = lo - pad
    hi = hi + pad
    return [
        lo[0].item(), hi[0].item(),
        lo[1].item(), hi[1].item(),
        lo[2].item(), hi[2].item(),
    ]


def estimate_volumes_occupancy(pos: torch.Tensor, n_grid: int) -> torch.Tensor:
    """Per-particle volume ``dx^3 / cells_occupancy`` — correct for dense-filled clouds."""
    from physics_sim.preprocessing.particle_filling import get_particle_volume

    lo = pos.min(dim=0).values
    hi = pos.max(dim=0).values
    extent = (hi - lo).max().item()
    if extent < 1e-8:
        extent = 1.0
    dx = extent / max(n_grid, 1)
    vol = get_particle_volume(pos - lo, grid_n=n_grid, grid_dx=dx)
    return vol.to(pos.device)


def _group_objects(sim_objects: list[SceneObject]) -> list[list[int]]:
    """Partition indices by ``fill_group``.  None → single-member group."""
    groups: list[list[int]] = []
    name_to_group: dict[str, int] = {}
    for i, o in enumerate(sim_objects):
        if o.fill_group is None:
            groups.append([i])
            continue
        if o.fill_group not in name_to_group:
            name_to_group[o.fill_group] = len(groups)
            groups.append([])
        groups[name_to_group[o.fill_group]].append(i)
    return groups


def _pick_group_config(
    members: list[SceneObject], group_idx: int,
) -> FillingConfig | None:
    """Return the shared FillingConfig for a group, or None if nobody set one."""
    filled = [m for m in members if m.particle_filling is not None]
    if not filled:
        return None
    first = filled[0].particle_filling
    for m in filled[1:]:
        if m.particle_filling != first:
            raise configuration_error(
                owner="scene.filling",
                operation="apply_particle_filling",
                expected="one FillingConfig per fill_group",
                detail=f"group #{group_idx}: part {m.name!r} differs from {filled[0].name!r}",
            )
    return first


# ── Group-level filling ─────────────────────────────────────────────────

def _concat_group_sources(members: list[SceneObject]) -> dict[str, torch.Tensor]:
    """Concatenate per-part tensors + synthetic part_id for the group."""
    device = members[0].positions.device
    part_ids = [
        torch.full((m.n_particles,), i, dtype=torch.long, device=device)
        for i, m in enumerate(members)
    ]
    return {
        "pos": torch.cat([m.positions for m in members], dim=0),
        "opacity": torch.cat([m.opacities for m in members], dim=0),
        "cov": torch.cat([m.covariances for m in members], dim=0),
        "shs": torch.cat([m.shs for m in members], dim=0),
        "quats": torch.cat([m.quats for m in members], dim=0),
        "scales": torch.cat([m.scales for m in members], dim=0),
        "part_id": torch.cat(part_ids, dim=0),
    }


def _run_fill_particles(
    src: dict[str, torch.Tensor], cfg: FillingConfig,
) -> torch.Tensor:
    """Call the preprocessing filler and return just the *new* positions."""
    from physics_sim.preprocessing.particle_filling import fill_particles

    boundary = _derive_boundary(src["pos"])
    n_src = src["pos"].shape[0]
    combined = fill_particles(
        pos=src["pos"],
        opacity=src["opacity"].reshape(-1),
        cov=src["cov"],
        grid_n=cfg.n_grid,
        max_samples=cfg.max_particles_num,
        grid_dx=1.0,  # overridden inside fill_particles when boundary is set
        density_thres=cfg.density_threshold,
        search_thres=cfg.search_threshold,
        threshold_mode=cfg.threshold_mode,
        max_particles_per_cell=cfg.max_particles_per_cell,
        search_exclude_dir=cfg.search_exclude_direction,
        ray_cast_dir=cfg.ray_cast_direction,
        boundary=boundary,
        smooth=cfg.smooth,
        progress=False,
    )
    return combined[n_src:] if combined.shape[0] > n_src else combined[:0]


def _partition_back(
    members: list[SceneObject],
    new_pos: torch.Tensor,
    new_attrs: dict[str, torch.Tensor],
) -> list[SceneObject]:
    out: list[SceneObject] = []
    new_part_id = new_attrs["part_id"]
    for part_id, m in enumerate(members):
        mask = new_part_id == part_id
        if not mask.any():
            out.append(m)
            continue
        out.append(replace(
            m,
            positions=torch.cat([m.positions, new_pos[mask]], dim=0),
            covariances=torch.cat([m.covariances, new_attrs["cov"][mask]], dim=0),
            opacities=torch.cat([m.opacities, new_attrs["opacity"][mask]], dim=0),
            shs=torch.cat([m.shs, new_attrs["shs"][mask]], dim=0),
            quats=torch.cat([m.quats, new_attrs["quats"][mask]], dim=0),
            scales=torch.cat([m.scales, new_attrs["scales"][mask]], dim=0),
        ))
    return out


def _fill_group(
    members: list[SceneObject],
    cfg: FillingConfig,
    strategy: FillStrategy,
) -> list[SceneObject]:
    src = _concat_group_sources(members)
    new_pos = _run_fill_particles(src, cfg)
    if new_pos.shape[0] == 0:
        return list(members)
    src_attrs = {k: v for k, v in src.items() if k != "pos"}
    new_attrs = strategy.inherit(
        src_pos=src["pos"], src_attrs=src_attrs, new_pos=new_pos,
    )
    return _partition_back(members, new_pos, new_attrs)


# ── Entry point ─────────────────────────────────────────────────────────

def apply_particle_filling(
    sim_objects: list[SceneObject],
    strategy: FillStrategy | None = None,
) -> list[SceneObject]:
    """Run filling on any sim_object whose part declared ``particle_filling``.

    Parts without filling are passed through unchanged; groups are honored via
    :attr:`SceneObject.fill_group`.  The returned list mirrors the input in
    length and order.
    """
    if not any(o.particle_filling is not None for o in sim_objects):
        return list(sim_objects)

    strategy = strategy or NearestNeighborStrategy()
    out: list[SceneObject | None] = [None] * len(sim_objects)
    for group_idx, group in enumerate(_group_objects(sim_objects)):
        members = [sim_objects[i] for i in group]
        cfg = _pick_group_config(members, group_idx)
        updated = _fill_group(members, cfg, strategy) if cfg is not None else members
        for i, obj in zip(group, updated):
            out[i] = obj
    return [o for o in out if o is not None]


def fill_scene_objects(
    objects: list[SceneObject],
    strategy: FillStrategy | None = None,
) -> tuple[list[SceneObject], bool]:
    """Apply filling to the ``dynamic`` entries of ``objects`` and splice back.

    Returns the (possibly) updated list plus a ``filled`` flag so the caller
    can switch volume estimators accordingly.  When no dynamic part declared
    filling, the original list is returned with ``filled=False``.
    """
    dynamic = [o for o in objects if o.role == "dynamic"]
    if not any(o.particle_filling is not None for o in dynamic):
        return list(objects), False
    filled_dyn = apply_particle_filling(dynamic, strategy)
    it = iter(filled_dyn)
    return [next(it) if o.role == "dynamic" else o for o in objects], True
