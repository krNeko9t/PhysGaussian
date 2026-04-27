"""Data layer for particle-filling diagnostics.

Re-runs scene assembly + filling to capture pre/post-fill positions per
fill group, and computes counts / AABB / distance-to-source stats.

Plotting + report writing live in :mod:`filling_inspect_plots`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from physics_sim.config.models import SimConfig
from physics_sim.render.interfaces import SceneAssetLoader
from physics_sim.scene.assembler import assemble_scene
from physics_sim.scene.filling import _group_objects, apply_particle_filling

AXIS_LABELS = ("x", "y", "z")


@dataclass
class FillSnapshot:
    group_name: Optional[str]
    member_names: list[str]
    src_pos: torch.Tensor
    filled_pos: torch.Tensor
    grid_dx: float
    n_grid: int

    @property
    def n_source(self) -> int:
        return int(self.src_pos.shape[0])

    @property
    def n_filled(self) -> int:
        return int(self.filled_pos.shape[0])

    def display_name(self) -> str:
        if self.group_name:
            return self.group_name
        return self.member_names[0] if self.member_names else "fill"


def _derive_grid_dx(src_pos: torch.Tensor, n_grid: int) -> float:
    """Mirror ``_derive_boundary`` + ``fill_particles`` grid_dx computation."""
    lo = src_pos.min(dim=0).values
    hi = src_pos.max(dim=0).values
    max_side = (hi - lo).max().item()
    return max(max_side * 1.10 / max(n_grid, 1), 1e-12)


def _empty_like(ref: torch.Tensor) -> torch.Tensor:
    return torch.empty((0, 3), device=ref.device, dtype=ref.dtype)


def capture_fill_snapshots(
    cfg: SimConfig,
    loader: SceneAssetLoader,
    config_dir: str = "",
) -> list[FillSnapshot]:
    """Run assembly + filling, return one snapshot per active fill group."""
    objs = assemble_scene(cfg, loader, config_dir=config_dir)
    dynamic = [o for o in objs if o.role == "dynamic"]
    if not any(o.particle_filling is not None for o in dynamic):
        return []

    n_before = [o.n_particles for o in dynamic]
    filled_objs = apply_particle_filling(dynamic)

    snaps: list[FillSnapshot] = []
    for group in _group_objects(filled_objs):
        members = [filled_objs[i] for i in group]
        cfg_fill = next(
            (m.particle_filling for m in members if m.particle_filling is not None),
            None,
        )
        if cfg_fill is None:
            continue
        src_parts = [filled_objs[i].positions[: n_before[i]] for i in group]
        fill_parts = [filled_objs[i].positions[n_before[i]:] for i in group]
        src_pos = torch.cat(src_parts, dim=0)
        total_fill = sum(int(p.shape[0]) for p in fill_parts)
        filled_pos = (
            torch.cat(fill_parts, dim=0) if total_fill > 0 else _empty_like(src_pos)
        )
        snaps.append(FillSnapshot(
            group_name=members[0].fill_group,
            member_names=[m.name for m in members],
            src_pos=src_pos,
            filled_pos=filled_pos,
            grid_dx=_derive_grid_dx(src_pos, cfg_fill.n_grid),
            n_grid=cfg_fill.n_grid,
        ))
    return snaps


# ── Stats ─────────────────────────────────────────────────────────────────

def _nearest_src_dist(
    filled: torch.Tensor, src: torch.Tensor, chunk: int = 4096,
) -> torch.Tensor:
    if filled.shape[0] == 0 or src.shape[0] == 0:
        return torch.empty(filled.shape[0], device=filled.device, dtype=filled.dtype)
    out = torch.empty(filled.shape[0], device=filled.device, dtype=filled.dtype)
    for start in range(0, filled.shape[0], chunk):
        end = min(start + chunk, filled.shape[0])
        d = torch.cdist(filled[start:end], src)
        out[start:end] = d.min(dim=1).values
    return out


def aabb(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Min/max corner of an (N,3) point set."""
    return pos.min(dim=0).values, pos.max(dim=0).values


def compute_stats(
    snap: FillSnapshot, outlier_dist: Optional[float] = None,
) -> dict:
    src, fill = snap.src_pos, snap.filled_pos
    src_lo, src_hi = aabb(src)
    threshold = outlier_dist if outlier_dist is not None else 2.0 * snap.grid_dx
    if fill.shape[0] > 0:
        fill_lo, fill_hi = aabb(fill)
        inside = torch.all((fill >= src_lo) & (fill <= src_hi), dim=-1)
        outside_count = int((~inside).sum().item())
        dist = _nearest_src_dist(fill, src).detach().cpu().numpy()
        far_count = int((dist > threshold).sum())
        dist_stats = {
            "min": float(dist.min()),
            "mean": float(dist.mean()),
            "p95": float(np.percentile(dist, 95)),
            "max": float(dist.max()),
        }
    else:
        fill_lo = fill_hi = torch.zeros(3)
        outside_count = 0
        far_count = 0
        dist_stats = {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n_source": snap.n_source,
        "n_filled": snap.n_filled,
        "grid_dx": snap.grid_dx,
        "n_grid": snap.n_grid,
        "src_aabb": (src_lo.tolist(), src_hi.tolist()),
        "filled_aabb": (fill_lo.tolist(), fill_hi.tolist()),
        "filled_outside_src_aabb": outside_count,
        "outlier_threshold": threshold,
        "filled_far_from_src_count": far_count,
        "nearest_src_dist": dist_stats,
    }
