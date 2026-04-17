"""Pipeline orchestration with explicit headless/render branches."""

from __future__ import annotations

import os
from dataclasses import dataclass

from physics_sim.config.loader import load_config
from physics_sim.render.runtime import create_render_runtime, create_scene_asset_loader
from physics_sim.stages.backend_init import init_backend
from physics_sim.stages.camera_setup import setup_camera
from physics_sim.stages.runtime import init_runtime
from physics_sim.stages.scene_setup import SceneData, setup_scene
from physics_sim.stages.sim_loop import RenderArgs, run_headless, run_with_rendering
from physics_sim.stages.video import compile_video


@dataclass
class PipelineRequest:
    config_path: str
    white_bg: bool = False
    compile_video: bool = False
    no_render: bool = False
    sh_degree: int = 3
    raster_backend: str = "gsplat"


class PipelineOrchestrator:
    """Thin orchestrator that wires stages and branch-specific runtimes."""

    def __init__(self, request: PipelineRequest):
        self.request = request
        self.cfg = None
        self.config_dir = ""
        self.scene_data: SceneData | None = None
        self.backend = None

    def prepare_scene(self) -> None:
        raw = self.request
        assert os.path.exists(raw.config_path), f"Config not found: {raw.config_path}"
        print("Loading config...")
        self.cfg = load_config(raw.config_path)
        self.config_dir = os.path.dirname(os.path.abspath(raw.config_path))
        os.makedirs(self.cfg.output, exist_ok=True)

        init_runtime(self.cfg.backend.type)
        loader = create_scene_asset_loader(sh_degree=raw.sh_degree)
        self.scene_data = setup_scene(
            self.cfg,
            loader,
            config_dir=self.config_dir,
            sh_degree=raw.sh_degree,
        )
        self.backend = init_backend(self.cfg, self.scene_data)

    def run_headless(self) -> None:
        assert self.cfg is not None and self.backend is not None and self.scene_data is not None
        run_headless(self.cfg, self.backend, self.scene_data)

    def run_rendering(self) -> None:
        assert self.cfg is not None and self.backend is not None and self.scene_data is not None
        raw = self.request
        runtime = create_render_runtime(
            sh_degree=raw.sh_degree,
            raster_backend=raw.raster_backend,
        )
        camera_state = setup_camera(self.cfg, self.scene_data, config_dir=self.config_dir)
        render_args = RenderArgs(white_bg=raw.white_bg)
        run_with_rendering(
            self.cfg,
            self.backend,
            self.scene_data,
            camera_state,
            runtime,
            render_args,
        )
        if raw.compile_video:
            compile_video(self.cfg.output, self.cfg.time.frame_dt, self.cfg.time.frame_num)

    def run(self) -> None:
        self.prepare_scene()
        if self.request.no_render:
            self.run_headless()
        else:
            self.run_rendering()
