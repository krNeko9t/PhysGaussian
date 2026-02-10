#!/usr/bin/env python3
"""
Standalone physics simulation + rendering pipeline for 3DGS scenes.

Usage:
    python pipeline.py \
        --ply_path  path/to/point_cloud.ply \
        --config    config/wolf_config.json  \
        --output    output/wolf              \
        --cameras_json path/to/cameras.json  \
        [--white_bg] [--compile_video]

Only depends on ``physics_sim/`` — no gaussian-splatting submodule needed.
"""

import argparse
import os
import cv2
import torch
import numpy as np
import warp as wp
import taichi as ti
from tqdm import tqdm

_debug_log_path = None


def _log(msg: str):
    """Write debug message either to log file (if set) or stdout."""
    global _debug_log_path
    if _debug_log_path is None:
        print(msg)
    else:
        os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
        with open(_debug_log_path, "a") as f:
            f.write(msg + "\n")


def _dbg(label: str, t: torch.Tensor):
    """Print summary statistics for a tensor (used in --debug mode)."""
    t_f = t.float()
    _log(
        f"  [DBG] {label:40s} shape={str(list(t.shape)):20s} "
        f"mean={t_f.mean().item():12.6f}  min={t_f.min().item():12.6f}  "
        f"max={t_f.max().item():12.6f}  sum={t_f.sum().item():14.4f}"
    )

# ── physics_sim modules ──────────────────────────────────────────────────
from physics_sim.config.parser import decode_param_json
from physics_sim.preprocessing.transform import (
    generate_rotation_matrices,
    apply_rotations,
    apply_cov_rotations,
    transform2origin,
    shift2center111,
    undoshift2center111,
    undotransform2origin,
    apply_inverse_rotations,
    apply_inverse_cov_rotations,
    get_center_view_worldspace_and_observant_coordinate,
)
from physics_sim.preprocessing.particle_filling import (
    fill_particles,
    get_particle_volume,
    init_filled_particles,
)
from physics_sim.backend.warp_mpm import WarpMPMBackend
from physics_sim.renderer.gs_renderer import GaussianRenderer


def main():
    parser = argparse.ArgumentParser(description="3DGS Physics Simulation Pipeline")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to a 3DGS point_cloud.ply file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the scene config JSON")
    parser.add_argument("--cameras_json", type=str, default=None,
                        help="Path to cameras.json (if not alongside the PLY)")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for frames and video")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree of the PLY model (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Print intermediate tensor statistics for debugging")
    parser.add_argument(
        "--debug_log",
        type=str,
        default=None,
        help="Path to a log file for debug output (used with --debug)",
    )
    args = parser.parse_args()

    # Infer cameras.json path: same directory as PLY by default
    if args.cameras_json is None:
        ply_dir = os.path.dirname(args.ply_path)
        # Walk up until we find cameras.json (typically at model root)
        candidate = ply_dir
        for _ in range(5):
            if os.path.exists(os.path.join(candidate, "cameras.json")):
                args.cameras_json = os.path.join(candidate, "cameras.json")
                break
            candidate = os.path.dirname(candidate)
        if args.cameras_json is None:
            raise FileNotFoundError(
                "cameras.json not found. Please specify --cameras_json explicitly."
            )

    assert os.path.exists(args.ply_path), f"PLY not found: {args.ply_path}"
    assert os.path.exists(args.config), f"Config not found: {args.config}"
    assert os.path.exists(args.cameras_json), f"cameras.json not found: {args.cameras_json}"
    os.makedirs(args.output, exist_ok=True)

    # Configure debug log file (if any)
    if args.debug:
        global _debug_log_path
        if args.debug_log is not None:
            _debug_log_path = args.debug_log
        else:
            _debug_log_path = os.path.join(args.output, "debug.log")
        # Truncate existing log
        os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
        with open(_debug_log_path, "w"):
            pass

    device = "cuda:0"

    # ── 0. Initialise runtime ────────────────────────────────────────
    wp.init()
    wp.config.verify_cuda = True
    ti.init(arch=ti.cuda, device_memory_GB=8.0)

    # ── 1. Load config ───────────────────────────────────────────────
    print("Loading scene config...")
    material_params, bc_params, time_params, preprocessing_params, camera_params = (
        decode_param_json(args.config)
    )

    # ── 2. Load 3DGS PLY ─────────────────────────────────────────────
    print("Loading 3DGS point cloud...")
    renderer = GaussianRenderer(sh_degree=args.sh_degree)
    params = renderer.load_ply(args.ply_path)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    if args.debug:
        _log("=== CHECKPOINT 1: After PLY loading ===")
        _dbg("init_pos", init_pos)
        _dbg("init_cov", init_cov)
        _dbg("init_opacity", init_opacity)
        _dbg("init_shs", init_shs)
        _log(f"  [DBG] Total particles: {init_pos.shape[0]}")

    # ── 3. Preprocessing ──────────────────────────────────────────────
    print("Preprocessing scene...")

    # 3a. Filter low-opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_shs = init_shs[mask, :]

    if args.debug:
        _log(f"=== CHECKPOINT 2: After opacity filter (threshold={preprocessing_params['opacity_threshold']}) ===")
        _log(f"  [DBG] Particles remaining: {init_pos.shape[0]}")
        _dbg("init_pos", init_pos)
        _dbg("init_cov", init_cov)

    # 3b. Rotate
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    # 3c. Select simulation area
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None, None, None, None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        area_mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool, device="cuda")
        for i in range(3):
            area_mask = torch.logical_and(area_mask, rotated_pos[:, i] > boundary[2 * i])
            area_mask = torch.logical_and(area_mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~area_mask, :]
        unselected_cov = init_cov[~area_mask, :]
        unselected_opacity = init_opacity[~area_mask, :]
        unselected_shs = init_shs[~area_mask, :]

        rotated_pos = rotated_pos[area_mask, :]
        init_cov = init_cov[area_mask, :]
        init_opacity = init_opacity[area_mask, :]
        init_shs = init_shs[area_mask, :]

    if args.debug:
        _log("=== CHECKPOINT 3: After sim_area selection ===")
        _log(f"  [DBG] Selected (sim) particles: {rotated_pos.shape[0]}")
        _dbg("rotated_pos (sim only)", rotated_pos)
        if unselected_pos is not None:
            _log(f"  [DBG] Unselected particles: {unselected_pos.shape[0]}")

    # 3d. Transform to MPM domain [0,2]^3
    transformed_pos, scale_origin, original_mean_pos = transform2origin(
        rotated_pos, preprocessing_params["scale"]
    )
    transformed_pos = shift2center111(transformed_pos)

    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        _log("=== CHECKPOINT 4: After transform to MPM domain ===")
        _log(f"  [DBG] scale_origin = {scale_origin.item():.10f}")
        _dbg("original_mean_pos", original_mean_pos)
        _dbg("transformed_pos", transformed_pos)
        _dbg("init_cov (rotated+scaled)", init_cov)

    # 3e. Particle filling
    gs_num = transformed_pos.shape[0]
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # 3f. Compute particle volumes
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        uniform=material_params["material"] == "sand",
    ).to(device=device)

    # 3g. Initialise filled-particle attributes
    if filling_params is not None and filling_params.get("visualize", False):
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num], init_shs, init_cov, init_opacity, mpm_init_pos[gs_num:]
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        _log("=== CHECKPOINT 5: Before physics init ===")
        _log(f"  [DBG] gs_num = {gs_num}")
        _log(f"  [DBG] total particles (gs+filled) = {mpm_init_pos.shape[0]}")
        _dbg("mpm_init_pos", mpm_init_pos)
        _dbg("mpm_init_vol", mpm_init_vol)
        _dbg("mpm_init_cov", mpm_init_cov)
        _dbg("shs (for render)", shs)
        _dbg("opacity (for render)", opacity)

    # ── 4. Initialise physics backend ─────────────────────────────────
    print("Initialising physics backend (Warp-MPM)...")
    backend = WarpMPMBackend(device=device)
    backend.initialize(
        mpm_init_pos, mpm_init_vol, mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    backend.set_material(material_params)
    backend.set_boundary_conditions(bc_params, time_params)
    backend.finalize()  # finalize_mu_lam — must be after set_boundary_conditions

    # ── 5. Camera setup ───────────────────────────────────────────────
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda()
    )
    viewpoint_center_worldspace, observant_coordinates = (
        get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )
    )

    # ── 6. Simulation + rendering loop ────────────────────────────────
    print("Running simulation and rendering...")
    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)

    bg_color = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None

    for frame in tqdm(range(frame_num), desc="Simulating"):
        # Build camera for this frame
        camera = renderer.build_camera_from_json(
            args.cameras_json,
            camera_params,
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            current_frame=frame,
        )
        rasterize = renderer.build_rasterizer(camera, bg_color)

        # Substep simulation
        for step in range(step_per_frame):
            backend.step(substep_dt, frame)

        # Export simulation state
        state = backend.get_state()
        pos = state.positions[:gs_num].to(device)
        cov3D = state.covariances[:gs_num].to(device)
        rot = state.rotations[:gs_num].to(device)

        if args.debug and frame == 0:
            _log("=== CHECKPOINT 6: Frame 0 — raw MPM state (before inverse transform) ===")
            _dbg("pos (from MPM)", pos)
            _dbg("cov3D (from MPM)", cov3D)
            _dbg("rot (from MPM)", rot)

        # Inverse transform back to world space
        pos = apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(pos), scale_origin, original_mean_pos
            ),
            rotation_matrices,
        )
        cov3D = cov3D / (scale_origin * scale_origin)
        cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)

        if args.debug and frame == 0:
            _log("=== CHECKPOINT 7: Frame 0 — after inverse transform (world space) ===")
            _dbg("pos (world)", pos)
            _dbg("cov3D (world)", cov3D)

        # Merge with unselected particles
        cur_opacity = opacity_render
        cur_shs = shs_render
        if preprocessing_params["sim_area"] is not None:
            pos = torch.cat([pos, unselected_pos], dim=0)
            cov3D = torch.cat([cov3D, unselected_cov], dim=0)
            cur_opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
            cur_shs = torch.cat([shs_render, unselected_shs], dim=0)

        if args.debug and frame == 0:
            _log("=== CHECKPOINT 8: Frame 0 — final render inputs ===")
            _dbg("pos (final)", pos)
            _dbg("cov3D (final)", cov3D)
            _dbg("cur_opacity", cur_opacity)
            _dbg("cur_shs", cur_shs)
            _log(f"  [DBG] Total render particles: {pos.shape[0]}")

        # Convert SH → RGB and rasterise
        colors_precomp = renderer.convert_sh(cur_shs, camera, pos, rot)
        screen_points = torch.zeros((pos.shape[0], 3), device="cuda")
        rendering, radii = rasterize(
            means3D=pos,
            means2D=screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=cur_opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D,
        )

        # Save frame
        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        if height is None or width is None:
            height = cv2_img.shape[0] // 2 * 2
            width = cv2_img.shape[1] // 2 * 2
        cv2.imwrite(
            os.path.join(args.output, f"{frame}".rjust(4, "0") + ".png"),
            255 * cv2_img,
        )

    # ── 7. Compile video ──────────────────────────────────────────────
    if args.compile_video:
        fps = int(1.0 / frame_dt)
        cmd = (
            f"ffmpeg -framerate {fps} -i {args.output}/%04d.png "
            f"-c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p "
            f"{args.output}/output.mp4"
        )
        print(f"Compiling video: {cmd}")
        os.system(cmd)
        print(f"Video saved to {args.output}/output.mp4")

    print("Done!")


if __name__ == "__main__":
    main()
