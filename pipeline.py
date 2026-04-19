#!/usr/bin/env python3
"""Physics simulation + rendering pipeline for 3DGS / 2DGS scenes.

Usage:
    python pipeline.py --config experiments/wolf_bread_rigid.py
    python pipeline.py --config experiments/wolf_bread_rigid.py --white_bg --compile_video
    python pipeline.py --config experiments/wolf_bread_rigid.py --no_render

Experiment configs are Python files that define a ``config`` variable
of type :class:`~physics_sim.config.models.SimConfig`.
"""

from __future__ import annotations

import argparse

from physics_sim.pipeline_orchestrator import PipelineOrchestrator, PipelineRequest


def main():
    parser = argparse.ArgumentParser(
        description="3DGS Physics Simulation Pipeline",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to Python experiment config (.py)",
    )
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument(
        "--no_render", action="store_true",
        help="Run physics only, skip rendering.",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        choices=range(0, 5),
        help="Spherical harmonics degree used by loader and colorizer.",
    )
    parser.add_argument(
        "--raster_backend", type=str, default="gsplat",
        choices=["gsplat", "diffrast"],
    )
    raw = parser.parse_args()

    req = PipelineRequest(
        config_path=raw.config,
        white_bg=raw.white_bg,
        compile_video=raw.compile_video,
        no_render=raw.no_render,
        sh_degree=raw.sh_degree,
        raster_backend=raw.raster_backend,
    )
    PipelineOrchestrator(req).run()


if __name__ == "__main__":
    main()
