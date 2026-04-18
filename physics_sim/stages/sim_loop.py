"""Stage 4: Simulation loop (headless and with rendering).

All positions coming out of the physics backend are in the internal
Y-up coordinate system.  No inverse rotation is needed for rendering
because the camera is also set up in Y-up.

SH evaluation requires view directions in the PLY-native coordinate
system, so ``alignment_inv`` (internal -> source) is passed to
``convert_sh`` for that purpose.
"""

from __future__ import annotations

import os
import glob as _glob
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from physics_sim.config.models import SimConfig
from physics_sim.logging_utils import get_logger
from physics_sim.render.interfaces import RenderRuntime
from physics_sim.render.registries import resolve_camera_for_mode

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.stages.camera_setup import CameraState
    from physics_sim.stages.scene_setup import SceneData


@dataclass
class RenderArgs:
    """Per-frame simulation/render loop options.

    SH degree and raster backend are fixed when :class:`~physics_sim.render.interfaces.RenderRuntime`
    is constructed; they are not repeated here.
    """

    white_bg: bool = False


def run_headless(
    cfg: SimConfig,
    backend: PhysicsBackend,
    scene_data: SceneData,
    device: str = "cuda:0",
) -> None:
    """Run simulation without rendering."""
    tc = cfg.time
    substep_dt = tc.substep_dt
    step_per_frame = int(tc.frame_dt / substep_dt)

    LOGGER.info("Running simulation (no rendering)...")
    for frame in tqdm(range(tc.frame_num), desc="Simulating"):
        for _ in range(step_per_frame):
            backend.step(substep_dt, frame)

    state = backend.get_state()
    gs_num = scene_data.gs_num
    pos = state.positions[:gs_num].to(device)
    cov3D = state.covariances[:gs_num].to(device)
    rot = state.rotations[:gs_num].to(device)

    out_path = os.path.join(cfg.output, "final_state.npz")
    np.savez_compressed(
        out_path,
        positions=pos.detach().cpu().numpy(),
        covariances=cov3D.detach().cpu().numpy(),
        rotations=rot.detach().cpu().numpy(),
    )
    LOGGER.info("Saved final state to %s", out_path)


def run_with_rendering(
    cfg: SimConfig,
    backend: PhysicsBackend,
    scene_data: SceneData,
    camera_state: CameraState,
    render_runtime: RenderRuntime,
    render_args: RenderArgs,
    device: str = "cuda:0",
) -> None:
    """Run simulation with per-frame rendering."""
    tc = cfg.time
    substep_dt = tc.substep_dt
    frame_dt = tc.frame_dt
    frame_num = tc.frame_num
    step_per_frame = int(frame_dt / substep_dt)
    output_dir = cfg.output

    cam_cfg = cfg.camera
    camera_mode = cam_cfg.camera_mode
    cameras_json = cam_cfg.cameras_json

    LOGGER.info("Running simulation and rendering...")
    LOGGER.info(
        "substep_dt=%.2e frame_dt=%.2e steps/frame=%s frames=%s",
        substep_dt,
        frame_dt,
        step_per_frame,
        frame_num,
    )

    # Clean up stale frame PNGs
    stale = sorted(_glob.glob(os.path.join(output_dir, "[0-9]*.png")))
    if stale:
        expected = {
            os.path.join(output_dir, f"{i:04d}.png") for i in range(frame_num)
        }
        to_remove = [p for p in stale if p not in expected]
        if to_remove:
            LOGGER.info("Removing stale frame PNGs: %s", len(to_remove))
            for p in to_remove:
                os.remove(p)

    bg_color = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if render_args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    gs_num = scene_data.gs_num
    opacity_render = scene_data.sim_opacity
    shs_render = scene_data.sim_shs
    has_static = len(scene_data.static_chunks) > 0
    alignment_inv = scene_data.alignment_inv
    source_axes = scene_data.source_axes

    height: Optional[int] = None
    width: Optional[int] = None

    for frame in tqdm(range(frame_num), desc="Simulating"):
        camera = resolve_camera_for_mode(
            camera_mode,
            camera_builder=render_runtime,
            cameras_json=cameras_json,
            camera_params=camera_state.camera_params,
            center_view_world_space=camera_state.viewpoint_center_worldspace,
            observant_coordinates=camera_state.observant_coordinates,
            current_frame=frame,
            source_axes=source_axes,
        )

        # Physics substeps
        for _ in range(step_per_frame):
            backend.step(substep_dt, frame)

        state = backend.get_state()
        pos = state.positions[:gs_num].to(device)
        cov3D = state.covariances[:gs_num].to(device)
        rot = state.rotations[:gs_num].to(device)

        # Diagnostics
        if hasattr(backend, "get_diagnostics"):
            diag = backend.get_diagnostics()
            for bd in diag["bodies"]:
                p, v = bd["pos"], bd["vel"]
                pdists = bd["plane_distances"]
                nan_flag = " *** NaN! ***" if bd["has_nan"] else ""
                dist_str = "  ".join(
                    f"plane{pi}={sd:+.4f}" for pi, sd in pdists
                )
                LOGGER.info(
                    "[DIAG] frame=%s name=%s pos=[%.4f,%.4f,%.4f] vel=[%.4f,%.4f,%.4f] %s%s",
                    frame,
                    bd["name"],
                    p[0],
                    p[1],
                    p[2],
                    v[0],
                    v[1],
                    v[2],
                    dist_str,
                    nan_flag,
                )

        # Sanitize bad particles
        nan_mask = ~torch.isfinite(pos).all(dim=1)
        extreme_mask = (pos.abs() > 100.0).any(dim=1)
        bad_mask = nan_mask | extreme_mask
        if bad_mask.any():
            n_nan = int(nan_mask.sum().item())
            n_ext = int((extreme_mask & ~nan_mask).sum().item())
            raise RuntimeError(
                "Simulation diverged with invalid particles: "
                f"frame={frame}, nan_or_inf={n_nan}, extreme={n_ext}"
            )

        # 2DGS quats/scales — already in Y-up, no inverse needed
        render_quats = render_scales = None
        if scene_data.gs_type == "2dgs" and state.quats is not None:
            render_quats = state.quats[:gs_num].to(device)
            render_scales = state.scales[:gs_num].to(device)

        # Combine with static geometry
        cur_opacity = opacity_render
        cur_shs = shs_render
        if has_static:
            pos = torch.cat([pos, scene_data.static_pos], dim=0)
            cov3D = torch.cat([cov3D, scene_data.static_cov], dim=0)
            cur_opacity = torch.cat([opacity_render, scene_data.static_opacity], dim=0)
            cur_shs = torch.cat([shs_render, scene_data.static_shs], dim=0)
            if render_quats is not None and scene_data.static_quats is not None:
                render_quats = torch.cat(
                    [render_quats, scene_data.static_quats], dim=0,
                )
                render_scales = torch.cat(
                    [render_scales, scene_data.static_scales], dim=0,
                )

        # SH -> RGB (alignment_inv transforms view dirs to PLY-native space)
        colors_precomp = render_runtime.convert_sh(
            cur_shs, camera, pos, rot,
            alignment_inv=alignment_inv,
        )
        if scene_data.gs_type == "2dgs" and render_quats is not None:
            rendering, _ = render_runtime.render(
                camera=camera,
                means=pos,
                colors=colors_precomp,
                opacities=cur_opacity,
                bg_color=bg_color,
                quats=render_quats,
                scales=render_scales,
            )
        else:
            rendering, _ = render_runtime.render(
                camera=camera,
                means=pos,
                colors=colors_precomp,
                opacities=cur_opacity,
                bg_color=bg_color,
                cov6=cov3D,
            )

        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        if height is None or width is None:
            height = cv2_img.shape[0] // 2 * 2
            width = cv2_img.shape[1] // 2 * 2
        cv2.imwrite(
            os.path.join(output_dir, f"{frame:04d}.png"),
            255 * cv2_img,
        )

    LOGGER.info("Rendering loop finished.")
