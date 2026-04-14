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
import os

from physics_sim.config.loader import load_config
from physics_sim.renderer.gs_renderer import GaussianRenderer
from physics_sim.stages.runtime import init_runtime
from physics_sim.stages.scene_setup import setup_scene
from physics_sim.stages.backend_init import init_backend
from physics_sim.stages.camera_setup import setup_camera
from physics_sim.stages.sim_loop import RenderArgs, run_headless, run_with_rendering
from physics_sim.stages.video import compile_video


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
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument(
        "--raster_backend", type=str, default="gsplat",
        choices=["gsplat", "diffrast"],
    )
    raw = parser.parse_args()

    assert os.path.exists(raw.config), f"Config not found: {raw.config}"

    print("Loading config...")
    cfg = load_config(raw.config)
    config_dir = os.path.dirname(os.path.abspath(raw.config))
    os.makedirs(cfg.output, exist_ok=True)

    init_runtime(cfg.backend.type)

    raster_be = "gsplat" if raw.no_render else raw.raster_backend
    renderer = GaussianRenderer(
        sh_degree=raw.sh_degree,
        raster_backend=raster_be,
    )

    scene_data = setup_scene(
        cfg, renderer,
        config_dir=config_dir,
        sh_degree=raw.sh_degree,
    )

    backend = init_backend(cfg, scene_data)

    if raw.no_render:
        run_headless(cfg, backend, scene_data)
    else:
        camera_state = setup_camera(cfg, scene_data, config_dir=config_dir)
        render_args = RenderArgs(
            white_bg=raw.white_bg,
            sh_degree=raw.sh_degree,
            raster_backend=raw.raster_backend,
        )
        run_with_rendering(
            cfg, backend, scene_data, camera_state,
            renderer, render_args,
        )
        if raw.compile_video:
            compile_video(cfg.output, cfg.time.frame_dt, cfg.time.frame_num)


if __name__ == "__main__":
    main()
