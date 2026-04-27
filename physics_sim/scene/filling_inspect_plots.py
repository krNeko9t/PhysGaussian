"""Matplotlib plotting + report writer for filling diagnostics.

Consumes :class:`FillSnapshot` from :mod:`filling_inspect` and writes
3-view scatter, axis-slice PNGs, and a text report.  No physics/render
dependencies — matplotlib + numpy + torch only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_sim.scene.filling_inspect import (
    AXIS_LABELS,
    FillSnapshot,
    aabb,
    compute_stats,
)


def _downsample(pos: torch.Tensor, max_points: int) -> np.ndarray:
    if pos.shape[0] == 0:
        return np.empty((0, max(pos.shape[-1], 2)), dtype=np.float32)
    if pos.shape[0] <= max_points:
        return pos.detach().cpu().numpy()
    idx = torch.randperm(pos.shape[0], device=pos.device)[:max_points]
    return pos[idx].detach().cpu().numpy()


def _draw_scatter_2d(
    ax, src_xy, fill_xy, aabb_xy,
    *, point_size: float, title: str, xlabel: str, ylabel: str,
) -> None:
    if src_xy.shape[0] > 0:
        ax.scatter(
            src_xy[:, 0], src_xy[:, 1],
            s=point_size, c="#888888", alpha=0.4, linewidths=0, label="source",
        )
    if fill_xy.shape[0] > 0:
        ax.scatter(
            fill_xy[:, 0], fill_xy[:, 1],
            s=point_size, c="#d62728", alpha=0.5, linewidths=0, label="filled",
        )
    if aabb_xy is not None:
        (xmin, ymin), (xmax, ymax) = aabb_xy
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
            color="#2ca02c", linestyle="--", linewidth=1.0,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=7, markerscale=3)


def plot_three_view(
    snap: FillSnapshot, out_path: Path,
    *, max_points: int, point_size: float,
) -> None:
    src = _downsample(snap.src_pos, max_points)
    fill = _downsample(snap.filled_pos, max_points)
    src_lo, src_hi = aabb(snap.src_pos)
    lo = src_lo.detach().cpu().numpy()
    hi = src_hi.detach().cpu().numpy()
    views = [
        (0, 1, "XY (top)", "x", "y"),
        (0, 2, "XZ (front)", "x", "z"),
        (1, 2, "YZ (side)", "y", "z"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j, title, xl, yl) in zip(axes, views):
        src_ij = src[:, [i, j]] if src.shape[0] else src
        fill_ij = fill[:, [i, j]] if fill.shape[0] else fill
        _draw_scatter_2d(
            ax, src_ij, fill_ij,
            ((lo[i], lo[j]), (hi[i], hi[j])),
            point_size=point_size,
            title=f"{snap.display_name()} — {title}", xlabel=xl, ylabel=yl,
        )
    fig.suptitle(
        f"source={snap.n_source}  filled={snap.n_filled}  "
        f"grid_dx={snap.grid_dx:.4g}",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _slice_mask(pos: torch.Tensor, axis: int, lo: float, hi: float) -> torch.Tensor:
    return (pos[:, axis] >= lo) & (pos[:, axis] < hi)


def plot_slices(
    snap: FillSnapshot, axis: int, out_path: Path,
    *, n_slices: int, max_points: int, point_size: float,
) -> None:
    src_lo, src_hi = aabb(snap.src_pos)
    lo_v = float(src_lo[axis].item())
    hi_v = float(src_hi[axis].item())
    thickness = (hi_v - lo_v) * 0.03 if hi_v > lo_v else snap.grid_dx
    centers = np.linspace(lo_v, hi_v, n_slices + 2)[1:-1]
    other = [i for i in range(3) if i != axis]
    aabb_xy = (
        (float(src_lo[other[0]]), float(src_lo[other[1]])),
        (float(src_hi[other[0]]), float(src_hi[other[1]])),
    )
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4), squeeze=False)
    for ax, center in zip(axes[0], centers):
        lo_s, hi_s = center - thickness / 2, center + thickness / 2
        src_mask = _slice_mask(snap.src_pos, axis, lo_s, hi_s)
        fill_mask = _slice_mask(snap.filled_pos, axis, lo_s, hi_s)
        src_xy = _downsample(snap.src_pos[src_mask][:, other], max_points)
        fill_xy = _downsample(snap.filled_pos[fill_mask][:, other], max_points)
        title = (
            f"{AXIS_LABELS[axis]}={center:.3g}  "
            f"src={src_xy.shape[0]} fill={fill_xy.shape[0]}"
        )
        _draw_scatter_2d(
            ax, src_xy, fill_xy, aabb_xy,
            point_size=point_size, title=title,
            xlabel=AXIS_LABELS[other[0]], ylabel=AXIS_LABELS[other[1]],
        )
    fig.suptitle(
        f"{snap.display_name()} — slices along {AXIS_LABELS[axis]} "
        f"(thickness={thickness:.3g})",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fmt_aabb(box) -> str:
    lo, hi = box
    return (
        f"lo=({lo[0]:.3g}, {lo[1]:.3g}, {lo[2]:.3g})  "
        f"hi=({hi[0]:.3g}, {hi[1]:.3g}, {hi[2]:.3g})"
    )


def _fmt_stats_block(name: str, members: list[str], stats: dict) -> str:
    d = stats["nearest_src_dist"]
    return "\n".join([
        f"# {name}  members={members}",
        f"  source particles : {stats['n_source']}",
        f"  filled particles : {stats['n_filled']}",
        f"  n_grid           : {stats['n_grid']}",
        f"  grid_dx          : {stats['grid_dx']:.6g}",
        f"  source AABB      : {_fmt_aabb(stats['src_aabb'])}",
        f"  filled AABB      : {_fmt_aabb(stats['filled_aabb'])}",
        f"  filled OUTSIDE source AABB : {stats['filled_outside_src_aabb']}",
        f"  nearest-src dist  min={d['min']:.4g} mean={d['mean']:.4g} "
        f"p95={d['p95']:.4g} max={d['max']:.4g}",
        f"  filled far from src (> {stats['outlier_threshold']:.4g}) : "
        f"{stats['filled_far_from_src_count']}",
        "",
    ])


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def write_inspection_report(
    snapshots: list[FillSnapshot],
    out_dir: Path,
    *,
    n_slices: int,
    outlier_dist: Optional[float],
    max_points: int,
    point_size: float,
) -> list[tuple[str, dict]]:
    """Write report.txt + per-group PNGs. Returns [(name, stats), ...]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = [f"filling inspection — {len(snapshots)} group(s)\n"]
    summary: list[tuple[str, dict]] = []
    for snap in snapshots:
        stats = compute_stats(snap, outlier_dist=outlier_dist)
        name = snap.display_name()
        safe = _safe_name(name)
        plot_three_view(
            snap, out_dir / f"{safe}_3view.png",
            max_points=max_points, point_size=point_size,
        )
        for axis in range(3):
            plot_slices(
                snap, axis,
                out_dir / f"{safe}_slices_{AXIS_LABELS[axis]}.png",
                n_slices=n_slices, max_points=max_points, point_size=point_size,
            )
        report_lines.append(_fmt_stats_block(name, snap.member_names, stats))
        summary.append((name, stats))
    (out_dir / "report.txt").write_text("\n".join(report_lines))
    return summary
