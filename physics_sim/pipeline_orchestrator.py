"""Pipeline orchestration with explicit headless/render branches."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from physics_sim.config.loader import load_config
from physics_sim.errors import lifecycle_error
from physics_sim.logging_utils import get_logger
from physics_sim.render.runtime import create_render_runtime, create_scene_asset_loader
from physics_sim.stages.backend_init import init_backend
from physics_sim.stages.camera_setup import setup_camera
from physics_sim.stages.runtime import init_runtime
from physics_sim.stages.scene_setup import SceneData, setup_scene
from physics_sim.stages.sim_loop import RenderArgs, run_headless, run_with_rendering
from physics_sim.stages.video import compile_video

LOGGER = get_logger(__name__)
_DEBUG_LOG_PATH = "/mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/PhysGaussian/.cursor/debug-e9c2ff.log"
_DEBUG_SESSION_ID = "e9c2ff"


def _agent_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


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
        if not os.path.exists(raw.config_path):
            raise FileNotFoundError(f"Config not found: {raw.config_path}")
        LOGGER.info("Loading config: %s", raw.config_path)
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
        if self.cfg is None or self.backend is None or self.scene_data is None:
            raise lifecycle_error(
                owner="pipeline_orchestrator",
                operation="run_headless",
                expected="prepare_scene() must run first",
            )
        run_headless(self.cfg, self.backend, self.scene_data)

    def run_rendering(self) -> None:
        if self.cfg is None or self.backend is None or self.scene_data is None:
            raise lifecycle_error(
                owner="pipeline_orchestrator",
                operation="run_rendering",
                expected="prepare_scene() must run first",
            )
        raw = self.request
        runtime = create_render_runtime(
            sh_degree=raw.sh_degree,
            raster_backend=raw.raster_backend,
        )
        camera_state = setup_camera(self.cfg, self.scene_data, config_dir=self.config_dir)
        render_args = RenderArgs(white_bg=raw.white_bg)
        # region agent log
        _agent_debug_log(
            run_id="run2",
            hypothesis_id="N1",
            location="physics_sim/pipeline_orchestrator.py:run_rendering:before_sim_loop",
            message="run_with_rendering about to start",
            data={
                "compile_video_flag": bool(raw.compile_video),
                "frame_num": int(self.cfg.time.frame_num),
                "output_dir": self.cfg.output,
            },
        )
        # endregion
        run_with_rendering(
            self.cfg,
            self.backend,
            self.scene_data,
            camera_state,
            runtime,
            render_args,
        )
        # region agent log
        _agent_debug_log(
            run_id="run2",
            hypothesis_id="N2",
            location="physics_sim/pipeline_orchestrator.py:run_rendering:after_sim_loop",
            message="run_with_rendering returned",
            data={"compile_video_flag": bool(raw.compile_video)},
        )
        # endregion
        if raw.compile_video:
            # region agent log
            _agent_debug_log(
                run_id="run2",
                hypothesis_id="N3",
                location="physics_sim/pipeline_orchestrator.py:run_rendering:before_compile_video",
                message="compile_video about to start",
                data={"output_dir": self.cfg.output},
            )
            # endregion
            compile_video(self.cfg.output, self.cfg.time.frame_dt, self.cfg.time.frame_num)
            # region agent log
            _agent_debug_log(
                run_id="run2",
                hypothesis_id="N4",
                location="physics_sim/pipeline_orchestrator.py:run_rendering:after_compile_video",
                message="compile_video returned",
                data={"output_dir": self.cfg.output},
            )
            # endregion

    def run(self) -> None:
        self.prepare_scene()
        if self.request.no_render:
            self.run_headless()
        else:
            self.run_rendering()
