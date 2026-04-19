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
from typing import TYPE_CHECKING

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


@dataclass
class _DynamicStateSlice:
    """Per-frame dynamic tensors for the simulated particles."""

    positions: torch.Tensor
    covariances: torch.Tensor
    rotations: torch.Tensor
    quats: torch.Tensor | None
    scales: torch.Tensor | None


def _advance_simulation_substeps(
    backend: PhysicsBackend,
    *,
    substep_dt: float,
    step_per_frame: int,
    frame: int,
) -> None:
    for _ in range(step_per_frame):
        backend.step(substep_dt, frame)


def _slice_dynamic_state(
    backend: PhysicsBackend,
    *,
    gs_num: int,
    gs_type: str,
    device: str,
) -> _DynamicStateSlice:
    state = backend.get_state()
    quats = scales = None
    if gs_type == "2dgs" and state.quats is not None and state.scales is not None:
        quats = state.quats[:gs_num].to(device)
        scales = state.scales[:gs_num].to(device)
    return _DynamicStateSlice(
        positions=state.positions[:gs_num].to(device),
        covariances=state.covariances[:gs_num].to(device),
        rotations=state.rotations[:gs_num].to(device),
        quats=quats,
        scales=scales,
    )


def _log_backend_diagnostics(backend: PhysicsBackend, *, frame: int) -> None:
    diag = backend.get_diagnostics()
    if not diag:
        return
    bodies = diag.get("bodies", [])
    for bd in bodies:
        p, v = bd["pos"], bd["vel"]
        pdists = bd["plane_distances"]
        nan_flag = " *** NaN! ***" if bd["has_nan"] else ""
        dist_str = "  ".join(f"plane{pi}={sd:+.4f}" for pi, sd in pdists)
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


def _validate_particle_positions(positions: torch.Tensor, *, frame: int) -> None:
    nan_mask = ~torch.isfinite(positions).all(dim=1)
    extreme_mask = (positions.abs() > 100.0).any(dim=1)
    bad_mask = nan_mask | extreme_mask
    if not bad_mask.any():
        return
    n_nan = int(nan_mask.sum().item())
    n_ext = int((extreme_mask & ~nan_mask).sum().item())
    raise RuntimeError(
        "Simulation diverged with invalid particles: "
        f"frame={frame}, nan_or_inf={n_nan}, extreme={n_ext}"
    )


def _compose_render_inputs(
    scene_data: SceneData,
    dynamic_state: _DynamicStateSlice,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    positions = dynamic_state.positions
    covariances = dynamic_state.covariances
    opacities = scene_data.sim_opacity
    shs = scene_data.sim_shs
    quats = dynamic_state.quats
    scales = dynamic_state.scales

    has_static = len(scene_data.static_chunks) > 0
    if has_static:
        positions = torch.cat([positions, scene_data.static_pos], dim=0)
        covariances = torch.cat([covariances, scene_data.static_cov], dim=0)
        opacities = torch.cat([scene_data.sim_opacity, scene_data.static_opacity], dim=0)
        shs = torch.cat([scene_data.sim_shs, scene_data.static_shs], dim=0)
        if quats is not None and scene_data.static_quats is not None:
            quats = torch.cat([quats, scene_data.static_quats], dim=0)
            scales = torch.cat([scales, scene_data.static_scales], dim=0)
    return positions, covariances, dynamic_state.rotations, opacities, shs, quats, scales


def _write_frame_png(*, rendering: torch.Tensor, output_dir: str, frame: int) -> None:
    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        os.path.join(output_dir, f"{frame:04d}.png"),
        255 * cv2_img,
    )


def _cleanup_stale_frames(*, output_dir: str, frame_num: int) -> None:
    stale = sorted(_glob.glob(os.path.join(output_dir, "[0-9]*.png")))
    if not stale:
        return
    expected = {os.path.join(output_dir, f"{i:04d}.png") for i in range(frame_num)}
    to_remove = [p for p in stale if p not in expected]
    if not to_remove:
        return
    LOGGER.info("Removing stale frame PNGs: %s", len(to_remove))
    for path in to_remove:
        os.remove(path)


def _save_final_state_npz(
    *,
    output_dir: str,
    positions: torch.Tensor,
    covariances: torch.Tensor,
    rotations: torch.Tensor,
) -> None:
    out_path = os.path.join(output_dir, "final_state.npz")
    np.savez_compressed(
        out_path,
        positions=positions.detach().cpu().numpy(),
        covariances=covariances.detach().cpu().numpy(),
        rotations=rotations.detach().cpu().numpy(),
    )
    LOGGER.info("Saved final state to %s", out_path)


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
        _advance_simulation_substeps(
            backend,
            substep_dt=substep_dt,
            step_per_frame=step_per_frame,
            frame=frame,
        )

    state = _slice_dynamic_state(
        backend,
        gs_num=scene_data.gs_num,
        gs_type=scene_data.gs_type,
        device=device,
    )
    _save_final_state_npz(
        output_dir=cfg.output,
        positions=state.positions,
        covariances=state.covariances,
        rotations=state.rotations,
    )


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

    _cleanup_stale_frames(output_dir=output_dir, frame_num=frame_num)

    bg_color = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if render_args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    alignment_inv = scene_data.alignment_inv
    source_axes = scene_data.source_axes

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

        _advance_simulation_substeps(
            backend,
            substep_dt=substep_dt,
            step_per_frame=step_per_frame,
            frame=frame,
        )

        dynamic_state = _slice_dynamic_state(
            backend,
            gs_num=scene_data.gs_num,
            gs_type=scene_data.gs_type,
            device=device,
        )
        _log_backend_diagnostics(backend, frame=frame)
        _validate_particle_positions(dynamic_state.positions, frame=frame)

        (
            pos,
            cov3D,
            rot,
            cur_opacity,
            cur_shs,
            render_quats,
            render_scales,
        ) = _compose_render_inputs(scene_data, dynamic_state)

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

        _write_frame_png(
            rendering=rendering,
            output_dir=output_dir,
            frame=frame,
        )

    LOGGER.info("Rendering loop finished.")
