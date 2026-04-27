#!/usr/bin/env python3
"""Filling-inspection CLI: dumps 3-view PNGs, slice PNGs, and a text report
for every fill group in a SimConfig.  Purely diagnostic — does not start
the physics backend or rasterizer.

Usage::

    python scripts/inspect_filling.py \\
        --config experiments/alocasia_external_cam_mpm.py \\
        --out-dir /tmp/fill_debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `physics_sim` is importable when the
# script is invoked directly (mirrors how pipeline.py behaves via conftest).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics_sim.config.loader import load_config
from physics_sim.render.registries import create_asset_loader
from physics_sim.scene.filling_inspect import capture_fill_snapshots
from physics_sim.scene.filling_inspect_plots import write_inspection_report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect particle filling output.")
    p.add_argument("--config", required=True, help="Path to experiment config (.py).")
    p.add_argument("--out-dir", default=None, help="Output directory (default fill_debug/<stem>).")
    p.add_argument("--sh-degree", type=int, default=3, choices=range(0, 5))
    p.add_argument("--n-slices", type=int, default=5)
    p.add_argument("--outlier-dist", type=float, default=None,
                   help="Distance threshold for 'far-from-source' count (default 2*grid_dx).")
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--max-points-per-plot", type=int, default=200_000)
    return p.parse_args()


def _default_out_dir(config_path: str) -> Path:
    return Path("fill_debug") / Path(config_path).stem


def _print_summary(summary: list[tuple[str, dict]], out_dir: Path) -> None:
    print(f"\n[inspect_filling] wrote {len(summary)} group(s) to {out_dir}")
    for name, stats in summary:
        d = stats["nearest_src_dist"]
        print(
            f"  - {name}: source={stats['n_source']} filled={stats['n_filled']} "
            f"outside_aabb={stats['filled_outside_src_aabb']} "
            f"far={stats['filled_far_from_src_count']} "
            f"dist[min/mean/p95/max]={d['min']:.3g}/{d['mean']:.3g}/"
            f"{d['p95']:.3g}/{d['max']:.3g}"
        )


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    config_dir = str(Path(args.config).resolve().parent)
    loader = create_asset_loader("ply", sh_degree=args.sh_degree)

    snapshots = capture_fill_snapshots(cfg, loader, config_dir=config_dir)
    if not snapshots:
        print("[inspect_filling] no parts declare particle_filling — nothing to inspect.")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args.config)
    summary = write_inspection_report(
        snapshots, out_dir,
        n_slices=args.n_slices,
        outlier_dist=args.outlier_dist,
        max_points=args.max_points_per_plot,
        point_size=args.point_size,
    )
    _print_summary(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
