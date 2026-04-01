#!/usr/bin/env python3
"""
Standalone physics simulation + rendering pipeline for 3DGS scenes.

Usage:
    python pipeline.py \
        --ply_path  path/to/point_cloud.ply \
        --config    config/wolf_config.json  \
        --output    output/wolf              \
        --cameras_json path/to/cameras.json  \
        [--raster_backend gsplat|diffrast]   \
        [--white_bg] [--compile_video]

Only depends on ``physics_sim/`` — no gaussian-splatting submodule needed.
Rendering uses gsplat by default; pass ``--raster_backend diffrast`` for the
original diff_gaussian_rasterization backend.
"""

import argparse
import os
import cv2
import torch
import numpy as np
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


def apply_axis_permutation(
    pos: torch.Tensor, cov: torch.Tensor, perm: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply axis permutation and optional flip to positions and covariances.

    Args:
        pos: (N, 3) positions
        cov: (N, 6) upper-triangle covariance [c00, c01, c02, c11, c12, c22]
        perm: axis permutation string with optional negation, e.g. "xyz", "-yxz", "y-x-z"

    Returns:
        Transformed (pos, cov)
    """
    if perm == "xyz":
        return pos, cov

    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = []
    signs = []
    negate_next = False
    for c in perm.lower():
        if c == "-":
            negate_next = True
        elif c in axis_map:
            idx.append(axis_map[c])
            signs.append(-1.0 if negate_next else 1.0)
            negate_next = False

    signs_t = torch.tensor(signs, device=pos.device, dtype=pos.dtype)

    pos_new = pos[:, idx] * signs_t

    cov_full = torch.zeros((pos.shape[0], 3, 3), device=cov.device)
    cov_full[:, 0, 0] = cov[:, 0]
    cov_full[:, 0, 1] = cov_full[:, 1, 0] = cov[:, 1]
    cov_full[:, 0, 2] = cov_full[:, 2, 0] = cov[:, 2]
    cov_full[:, 1, 1] = cov[:, 3]
    cov_full[:, 1, 2] = cov_full[:, 2, 1] = cov[:, 4]
    cov_full[:, 2, 2] = cov[:, 5]

    cov_perm = cov_full[:, idx, :][:, :, idx]

    sign_matrix = signs_t.unsqueeze(0) * signs_t.unsqueeze(1)
    cov_perm = cov_perm * sign_matrix

    cov_new = torch.stack(
        [
            cov_perm[:, 0, 0],
            cov_perm[:, 0, 1],
            cov_perm[:, 0, 2],
            cov_perm[:, 1, 1],
            cov_perm[:, 1, 2],
            cov_perm[:, 2, 2],
        ],
        dim=1,
    )

    return pos_new, cov_new


def apply_axis_perm_to_quats(quats_wxyz: torch.Tensor, perm: str) -> torch.Tensor:
    """Rotate quaternions to account for axis permutation.

    Positions get axis-permuted via ``apply_axis_permutation``, so the
    Gaussian orientations must be rotated by the same permutation matrix
    to stay consistent: ``q_new = quat(P) * q_old``.
    """
    if perm == "xyz":
        return quats_wxyz
    device = quats_wxyz.device
    P = _build_axis_perm_matrix(perm, device)
    if torch.det(P) < 0:
        P = -P
    q_perm = _rotmat_to_quat_wxyz(P)
    return _quat_mul_wxyz(q_perm, quats_wxyz)


def _build_axis_perm_matrix(perm: str, device: torch.device) -> torch.Tensor:
    """Build a 3×3 signed permutation matrix from an axis permutation string."""
    if perm == "xyz":
        return torch.eye(3, device=device, dtype=torch.float32)
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx, signs = [], []
    negate_next = False
    for c in perm.lower():
        if c == "-":
            negate_next = True
        elif c in axis_map:
            idx.append(axis_map[c])
            signs.append(-1.0 if negate_next else 1.0)
            negate_next = False
    P = torch.zeros(3, 3, device=device, dtype=torch.float32)
    for row, (col, s) in enumerate(zip(idx, signs)):
        P[row, col] = s
    return P


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix → (4,) quaternion (w, x, y, z).  Batched (N, 3, 3) → (N, 4) also supported."""
    single = R.dim() == 2
    if single:
        R = R.unsqueeze(0)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)
    # Case 1: trace > 0
    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(trace[m1] + 1.0) * 2.0
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s
    # Case 2: R[0,0] max
    m2 = ~m1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = torch.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2.0
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s
    # Case 3: R[1,1] max
    m3 = ~m1 & ~m2 & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = torch.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2.0
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s
    # Case 4: R[2,2] max
    m4 = ~m1 & ~m2 & ~m3
    if m4.any():
        s = torch.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]) * 2.0
        q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 3] = 0.25 * s
    q = torch.nn.functional.normalize(q, dim=-1)
    return q[0] if single else q


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (w, x, y, z) quaternions.  Supports (4,)*(N,4) → (N,4)."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    aw, ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bw, bx, by, bz = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:4]
    return torch.cat([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def preprocess_quats(
    quats_wxyz: torch.Tensor,
    axis_perm: str,
    rotation_matrices: list[torch.Tensor],
) -> torch.Tensor:
    """Apply the same preprocessing rotation chain (axis perm + rotation matrices) to quaternions.

    Returns quaternions in MPM-space (wxyz convention).
    """
    device = quats_wxyz.device
    # Build combined rotation matrix
    R_combined = _build_axis_perm_matrix(axis_perm, device)
    if torch.det(R_combined) < 0:
        R_combined = -R_combined
    for R in rotation_matrices:
        R_combined = R @ R_combined
    q_pre = _rotmat_to_quat_wxyz(R_combined)
    return _quat_mul_wxyz(q_pre, quats_wxyz)


def inverse_preprocess_quats(
    quats_mpm_wxyz: torch.Tensor,
    axis_perm: str,
    rotation_matrices: list[torch.Tensor],
) -> torch.Tensor:
    """Undo the preprocessing rotation on quaternions (MPM → world)."""
    device = quats_mpm_wxyz.device
    R_combined = _build_axis_perm_matrix(axis_perm, device)
    if torch.det(R_combined) < 0:
        R_combined = -R_combined
    for R in rotation_matrices:
        R_combined = R @ R_combined
    q_pre = _rotmat_to_quat_wxyz(R_combined)
    q_pre_inv = q_pre.clone()
    q_pre_inv[1:] = -q_pre_inv[1:]  # conjugate = inverse for unit quat
    return _quat_mul_wxyz(q_pre_inv, quats_mpm_wxyz)


# ── physics_sim modules ──────────────────────────────────────────────────
from physics_sim.config.parser import decode_param_json
from physics_sim.preprocessing.transform import (
    generate_rotation_matrices,
    apply_rotations,
    apply_cov_rotations,
    transform2origin,
    transform_with_reference,
    shift2center111,
    undoshift2center111,
    undotransform2origin,
    apply_inverse_rotations,
    apply_inverse_cov_rotations,
    get_center_view_worldspace_and_observant_coordinate,
    generate_local_coord,
    world_to_mpm_positions,
    world_to_mpm_directions,
    mpm_to_world_positions,
)
from physics_sim.backend.base import SimulationState
from physics_sim.renderer.gs_renderer import GaussianRenderer
from physics_sim.geometry.plane_fit import fit_plane_svd


def main():
    parser = argparse.ArgumentParser(description="3DGS Physics Simulation Pipeline")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to a 3DGS point_cloud.ply file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the scene config JSON")
    parser.add_argument("--cameras_json", type=str, default=None,
                        help="Path to cameras.json (if not alongside the PLY)")
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="json",
        choices=["json", "orbit", "fixed"],
        help="Camera source: cameras.json (json) or procedural (orbit/fixed).",
    )
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for frames and video")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Run physics but skip rendering (no rasterization backend needed).",
    )
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree of the PLY model (default: 3)")
    parser.add_argument(
        "--raster_backend", type=str, default="gsplat",
        choices=["gsplat", "diffrast"],
        help="Rasterization backend: gsplat (default) or diffrast",
    )
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "none", "warp_mpm", "newton_mpm", "newton_rigid", "newton_vbd"],
                        help="Physics backend (default: auto — read from config JSON, fallback warp_mpm)")
    parser.add_argument("--debug", action="store_true",
                        help="Print intermediate tensor statistics for debugging")
    parser.add_argument(
        "--debug_log",
        type=str,
        default=None,
        help="Path to a log file for debug output (used with --debug)",
    )
    args = parser.parse_args()

    # Infer cameras.json path only when we use json cameras.
    if args.camera_mode == "json":
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
    if args.camera_mode == "json":
        assert os.path.exists(args.cameras_json), (
            f"cameras.json not found: {args.cameras_json}"
        )
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
    # Import heavy runtimes lazily so the script can still run in
    # render-only / config-debug scenarios (e.g. backend="none").
    wp = None
    ti = None
    if args.backend != "none":
        try:
            import warp as wp  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Python module 'warp' not found. Install the Warp package "
                "(pip package name: warp_lang) to run physics backends."
            ) from e
        try:
            import taichi as ti  # type: ignore
        except ModuleNotFoundError:
            ti = None

        wp.init()
        if args.backend in ("newton_mpm", "newton_rigid", "newton_vbd"):
            wp.config.verify_cuda = False  # Newton uses CUDA graph capture internally
        else:
            wp.config.verify_cuda = True

        # Taichi is only required by some legacy preprocessing paths.
        if ti is not None:
            ti.init(arch=ti.cuda, device_memory_GB=8.0)

    # ── 1. Load config ───────────────────────────────────────────────
    print("Loading scene config...")
    (material_params, bc_params, time_params,
     preprocessing_params, camera_params, backend_overrides,
     scene_objects, config_backend) = decode_param_json(args.config)

    # Resolve backend: CLI "auto" defers to the config JSON's "backend" field.
    if args.backend == "auto":
        if config_backend:
            args.backend = config_backend
            print(f"Backend from config: {args.backend}")
        else:
            args.backend = "warp_mpm"
            print("No backend in config, defaulting to warp_mpm")

    # Apply backend-specific config overrides (if the selected backend
    # has an override section in the JSON).  This keeps all tuneable
    # numbers in the config file instead of hardcoding them in Python.
    if args.backend in backend_overrides:
        overrides = backend_overrides[args.backend]
        for key in ("substep_dt", "frame_dt", "frame_num"):
            if key in overrides:
                old = time_params[key]
                time_params[key] = overrides[key]
                print(f"[{args.backend}] Override {key}: {old} -> {overrides[key]}")
        # Solver-specific options are passed through material_params
        # so the backend can pick them up in set_material().
        if "solver" in overrides:
            material_params["newton_solver_opts"] = overrides["solver"]

    # ── Optional Taichi-based preprocessing (particle filling / volumes) ──
    # Taichi is not required for rigid-body or render-only pipelines, and may
    # be unavailable in some environments. We import it only when needed.
    fill_particles = None
    get_particle_volume = None
    init_filled_particles = None

    needs_particle_filling = preprocessing_params.get("particle_filling") is not None
    if scene_objects is not None:
        needs_particle_filling = needs_particle_filling or any(
            (obj.get("particle_filling") is not None)
            for obj in scene_objects
            if obj.get("mode", "simulate") == "simulate"
        )

    # Backends that require per-particle volumes for mass computation.
    needs_volumes = args.backend in ("warp_mpm", "newton_mpm", "newton_vbd")

    if needs_particle_filling or needs_volumes:
        try:
            from physics_sim.preprocessing.particle_filling import (
                fill_particles as _fill_particles,
                get_particle_volume as _get_particle_volume,
                init_filled_particles as _init_filled_particles,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Taichi-based preprocessing is required by your configuration/backend, "
                "but Taichi is not installed. Install taichi or disable particle_filling "
                "and use a backend that doesn't require volumes (e.g. newton_rigid)."
            ) from e
        fill_particles = _fill_particles
        get_particle_volume = _get_particle_volume
        init_filled_particles = _init_filled_particles

    def _compute_volumes(pos: torch.Tensor, uniform: bool) -> torch.Tensor:
        if get_particle_volume is not None:
            return get_particle_volume(
                pos,
                material_params["n_grid"],
                material_params["grid_lim"] / material_params["n_grid"],
                uniform=uniform,
            ).to(device=device)
        # Fallback: uniform voxel volume (acceptable for rigid-only / render-only)
        dx = material_params["grid_lim"] / material_params["n_grid"]
        return torch.full((pos.shape[0],), float(dx * dx * dx), device=device)

    # ── 2. Load 3DGS / 2DGS PLY ────────────────────────────────────────
    print("Loading Gaussian point cloud...")
    raster_be = "gsplat" if args.no_render else args.raster_backend
    renderer = GaussianRenderer(sh_degree=args.sh_degree, raster_backend=raster_be)
    axis_perm = preprocessing_params.get("axis_permutation", "xyz")
    params = renderer.load_ply(args.ply_path)

    gs_type = params["gs_type"]
    print(f"  Detected PLY type: {gs_type.upper()}")

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_pos, init_cov = apply_axis_permutation(init_pos, init_cov, axis_perm)
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # 2DGS-specific: store original quats and scales for physics + rendering
    init_ply_quats = params["quats"]   # (N, 4) wxyz
    init_ply_scales = params["scales"] # (N, 2|3)

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
    init_ply_quats = init_ply_quats[mask, :]
    init_ply_scales = init_ply_scales[mask, :]

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

    # 3c–3g: Select simulation area(s), transform, fill, compute volumes.
    #
    # Two code paths:
    #   scene_objects is None  → single-object (backward compatible)
    #   scene_objects is list  → multi-object (per-object sim_area & material)
    #
    # Both paths produce the same output variables:
    #   mpm_init_pos, mpm_init_vol, mpm_init_cov  (all particles)
    #   gs_num                     (GS particle count, for rendering)
    #   shs, opacity               (GS render attributes)
    #   scale_origin, original_mean_pos  (coordinate transform params)
    #   static_pos/cov/opacity/shs/quats/scales (static particles, or None)
    #   has_static                 (bool, whether static particles exist)
    #   per_object_info            (None or list of per-object material+indices)

    # Static render-only particles (world space)
    static_pos = static_cov = static_opacity = static_shs = None
    static_quats = static_scales = None
    has_static = False

    # Multi-object mapping for physics backends
    per_object_info = None
    # Reference cloud (world space) used for procedural camera centering
    reference_world_pos = None
    # Additional boundary conditions generated from collider-only objects
    collider_bc_params = []

    if scene_objects is not None:
        # ── MULTI-OBJECT PATH ─────────────────────────────────────
        print(f"  Multi-object mode: {len(scene_objects)} objects")

        simulate_objects = [
            obj for obj in scene_objects if obj.get("mode", "simulate") == "simulate"
        ]
        render_only_objects = [
            obj for obj in scene_objects if obj.get("mode") == "render_only"
        ]
        collider_only_objects = [
            obj for obj in scene_objects if obj.get("mode") == "collider_only"
        ]

        # 3c(0). Load static particles for rendering (world space)
        static_chunks = []

        # 3c(0). Convert collider_only objects into surface colliders
        for obj in collider_only_objects:
            col = obj.get("collider") or {}
            col_type = col.get("type", "plane")
            if col_type != "plane":
                raise ValueError(
                    f"collider_only object '{obj.get('name', '?')}' only supports "
                    f"collider.type='plane' for now (got {col_type!r})"
                )

            space = col.get("space", "world")
            surface = col.get("surface", "sticky")
            friction = float(col.get("friction", 0.0))
            start_time = col.get("start_time", 0)
            end_time = col.get("end_time", 1e3)

            # By default, collider_only objects also render (if ply_path exists).
            # Can be disabled via collider.render=false (or object-level render=false).
            render_flag = col.get("render", None)
            if render_flag is None:
                render_flag = obj.get("render", True)
            render_flag = bool(render_flag)

            ply_params = None
            ply_path = obj.get("ply_path")
            if render_flag and ply_path is not None:
                print(f"    [collider_only/render] {obj['name']}: loading {ply_path}...")
                ply_params = renderer.load_ply(ply_path)
                r_pos = ply_params["pos"]
                r_cov = ply_params["cov3D_precomp"]
                r_pos, r_cov = apply_axis_permutation(r_pos, r_cov, axis_perm)
                r_opacity = ply_params["opacity"]
                r_shs = ply_params["shs"]
                op_mask = r_opacity[:, 0] > preprocessing_params["opacity_threshold"]
                static_chunks.append(
                    dict(
                        pos=r_pos[op_mask],
                        cov=r_cov[op_mask],
                        opacity=r_opacity[op_mask],
                        shs=r_shs[op_mask],
                        quats=ply_params["quats"][op_mask],
                        scales=ply_params["scales"][op_mask],
                    )
                )
                print(
                    f"    [collider_only/render] {obj['name']}: "
                    f"{int(op_mask.sum().item())} GS particles"
                )

            if col.get("point") is not None and col.get("normal") is not None:
                point = col["point"]
                normal = col["normal"]
            else:
                if ply_path is None:
                    raise ValueError(
                        f"collider_only object '{obj.get('name', '?')}' must provide "
                        "either collider.point+normal or ply_path for fitting."
                    )
                print(f"    [collider_only] {obj['name']}: fitting plane from {ply_path}...")
                if ply_params is None:
                    ply_params = renderer.load_ply(ply_path)
                c_pos = ply_params["pos"]
                c_cov = ply_params["cov3D_precomp"]
                c_pos, c_cov = apply_axis_permutation(c_pos, c_cov, axis_perm)
                c_opacity = ply_params["opacity"]
                op_mask = c_opacity[:, 0] > preprocessing_params["opacity_threshold"]
                pts = c_pos[op_mask].detach().cpu().numpy()

                fit = col.get("fit") or {}
                method = fit.get("method", "svd")
                if method != "svd":
                    raise ValueError(
                        f"collider_only plane fit only supports method='svd' for now "
                        f"(got {method!r})"
                    )
                sample_max = fit.get("sample_max", 200000)
                prefer_up = col.get("prefer_up", [0.0, 0.0, 1.0])
                res = fit_plane_svd(
                    pts,
                    sample_max=sample_max,
                    seed=int(fit.get("seed", 0)),
                    prefer_up=np.asarray(prefer_up, dtype=np.float32),
                )
                point = res.point.tolist()
                normal = res.normal.tolist()
                print(
                    f"    [collider_only] {obj['name']}: plane rms={res.rms:.6f}, "
                    f"point={point}, normal={normal}"
                )

            collider_bc_params.append(
                dict(
                    type="surface_collider",
                    space=space,
                    point=point,
                    normal=normal,
                    surface=surface,
                    friction=friction,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        # 3c(1). Load render-only PLYs (kept in world space)
        for obj in render_only_objects:
            ply_path = obj.get("ply_path")
            if ply_path is None:
                raise ValueError(
                    f"render_only object '{obj.get('name', '?')}' must define ply_path"
                )
            print(f"    [render_only] {obj['name']}: loading {ply_path}...")
            ply_params = renderer.load_ply(ply_path)
            r_pos = ply_params["pos"]
            r_cov = ply_params["cov3D_precomp"]
            r_pos, r_cov = apply_axis_permutation(r_pos, r_cov, axis_perm)
            r_opacity = ply_params["opacity"]
            r_shs = ply_params["shs"]
            op_mask = r_opacity[:, 0] > preprocessing_params["opacity_threshold"]
            static_chunks.append(
                dict(
                    pos=r_pos[op_mask],
                    cov=r_cov[op_mask],
                    opacity=r_opacity[op_mask],
                    shs=r_shs[op_mask],
                    quats=ply_params["quats"][op_mask],
                    scales=ply_params["scales"][op_mask],
                )
            )
            print(f"    [render_only] {obj['name']}: {int(op_mask.sum().item())} GS particles")

        # 3c(2). Collect simulate objects (rotated space for selection + transform)
        any_uses_shared = any(obj.get("ply_path") is None for obj in simulate_objects)
        all_selected = (
            torch.zeros(rotated_pos.shape[0], dtype=torch.bool, device="cuda")
            if any_uses_shared else None
        )
        obj_data = []

        for obj in simulate_objects:
            if obj.get("ply_path") is not None:
                ply_path = obj["ply_path"]
                print(f"    [simulate] {obj['name']}: loading {ply_path}...")
                ply_params = renderer.load_ply(ply_path)
                obj_pos = ply_params["pos"]
                obj_cov = ply_params["cov3D_precomp"]
                obj_pos, obj_cov = apply_axis_permutation(obj_pos, obj_cov, axis_perm)
                obj_opacity = ply_params["opacity"]
                obj_shs = ply_params["shs"]
                obj_quats = ply_params["quats"]
                obj_scales = ply_params["scales"]

                op_mask = obj_opacity[:, 0] > preprocessing_params["opacity_threshold"]
                obj_pos = obj_pos[op_mask]
                obj_cov = obj_cov[op_mask]
                obj_opacity = obj_opacity[op_mask]
                obj_shs = obj_shs[op_mask]
                obj_quats = obj_quats[op_mask]
                obj_scales = obj_scales[op_mask]

                obj_rotated_pos = apply_rotations(obj_pos, rotation_matrices)

                # Optional sim_area filter (in rotated space)
                if obj.get("sim_area") is not None:
                    boundary = obj["sim_area"]
                    assert len(boundary) == 6
                    sa_mask = torch.ones(
                        obj_rotated_pos.shape[0],
                        dtype=torch.bool,
                        device="cuda",
                    )
                    for i in range(3):
                        sa_mask = torch.logical_and(
                            sa_mask,
                            obj_rotated_pos[:, i] > boundary[2 * i],
                        )
                        sa_mask = torch.logical_and(
                            sa_mask,
                            obj_rotated_pos[:, i] < boundary[2 * i + 1],
                        )
                    obj_rotated_pos = obj_rotated_pos[sa_mask]
                    obj_cov = obj_cov[sa_mask]
                    obj_opacity = obj_opacity[sa_mask]
                    obj_shs = obj_shs[sa_mask]
                    obj_quats = obj_quats[sa_mask]
                    obj_scales = obj_scales[sa_mask]

                n_gs = obj_rotated_pos.shape[0]
                obj_data.append(
                    dict(
                        obj=obj,
                        rotated_pos=obj_rotated_pos,
                        cov=obj_cov,
                        opacity=obj_opacity,
                        shs=obj_shs,
                        quats=obj_quats,
                        scales=obj_scales,
                        gs_count=n_gs,
                    )
                )
                print(f"    [simulate] {obj['name']}: {n_gs} GS particles (dedicated PLY)")

            else:
                # Shared PLY selection by sim_area
                boundary = obj.get("sim_area")
                assert boundary is not None and len(boundary) == 6, (
                    f"Object '{obj['name']}' without ply_path must have sim_area[6]"
                )
                obj_mask = torch.ones(
                    rotated_pos.shape[0], dtype=torch.bool, device="cuda"
                )
                for i in range(3):
                    obj_mask = torch.logical_and(
                        obj_mask, rotated_pos[:, i] > boundary[2 * i]
                    )
                    obj_mask = torch.logical_and(
                        obj_mask, rotated_pos[:, i] < boundary[2 * i + 1]
                    )
                obj_mask = torch.logical_and(obj_mask, ~all_selected)
                all_selected = torch.logical_or(all_selected, obj_mask)

                n_gs = int(obj_mask.sum().item())
                obj_data.append(
                    dict(
                        obj=obj,
                        rotated_pos=rotated_pos[obj_mask],
                        cov=init_cov[obj_mask],
                        opacity=init_opacity[obj_mask],
                        shs=init_shs[obj_mask],
                        quats=init_ply_quats[obj_mask],
                        scales=init_ply_scales[obj_mask],
                        gs_count=n_gs,
                    )
                )
                print(f"    [simulate] {obj['name']}: {n_gs} GS particles (shared PLY)")

        if len(obj_data) == 0:
            raise ValueError("Multi-object mode requires at least one simulate object.")

        # Add unselected shared particles to static rendering (world space)
        if any_uses_shared:
            unsel_mask = ~all_selected
            if unsel_mask.any():
                static_chunks.append(
                    dict(
                        pos=init_pos[unsel_mask],
                        cov=init_cov[unsel_mask],
                        opacity=init_opacity[unsel_mask],
                        shs=init_shs[unsel_mask],
                        quats=init_ply_quats[unsel_mask],
                        scales=init_ply_scales[unsel_mask],
                    )
                )

        if static_chunks:
            static_pos = torch.cat([c["pos"] for c in static_chunks], dim=0)
            static_cov = torch.cat([c["cov"] for c in static_chunks], dim=0)
            static_opacity = torch.cat([c["opacity"] for c in static_chunks], dim=0)
            static_shs = torch.cat([c["shs"] for c in static_chunks], dim=0)
            static_quats = torch.cat([c["quats"] for c in static_chunks], dim=0)
            static_scales = torch.cat([c["scales"] for c in static_chunks], dim=0)
            has_static = True

        # Concatenate simulated GS particles
        rotated_pos_sim = torch.cat([d["rotated_pos"] for d in obj_data], dim=0)
        init_cov_sim = torch.cat([d["cov"] for d in obj_data], dim=0)
        init_opacity_sim = torch.cat([d["opacity"] for d in obj_data], dim=0)
        init_shs_sim = torch.cat([d["shs"] for d in obj_data], dim=0)
        init_ply_quats = torch.cat([d["quats"] for d in obj_data], dim=0)
        init_ply_scales = torch.cat([d["scales"] for d in obj_data], dim=0)

        # 3d. Compute reference transform (rotated space)
        ref_spec = preprocessing_params.get("transform_reference", None)
        ref_rotated_pos = None
        if isinstance(ref_spec, str):
            if ref_spec == "shared_ply":
                reference_world_pos = init_pos
                ref_rotated_pos = rotated_pos
            elif ref_spec in ("simulated", "", None):
                ref_rotated_pos = None
            else:
                raise ValueError(
                    f"Unknown transform_reference string: {ref_spec!r}. "
                    "Use 'shared_ply', 'simulated', or {'ply_path': ...}."
                )
        elif isinstance(ref_spec, dict) and ref_spec.get("ply_path") is not None:
            ref_ply = ref_spec["ply_path"]
            print(f"  Transform reference: loading {ref_ply}...")
            ref_params = renderer.load_ply(ref_ply)
            ref_pos = ref_params["pos"]
            ref_cov = ref_params["cov3D_precomp"]
            ref_pos, ref_cov = apply_axis_permutation(ref_pos, ref_cov, axis_perm)
            ref_opacity = ref_params["opacity"]
            op_mask = ref_opacity[:, 0] > preprocessing_params["opacity_threshold"]
            reference_world_pos = ref_pos[op_mask]
            ref_rotated_pos = apply_rotations(reference_world_pos, rotation_matrices)
        elif ref_spec is None:
            ref_rotated_pos = None
        else:
            raise ValueError(
                f"Invalid transform_reference: {ref_spec!r}. "
                "Expected None, 'shared_ply', 'simulated', or {'ply_path': ...}."
            )

        if ref_rotated_pos is None:
            ref_rotated_pos = rotated_pos_sim
            reference_world_pos = init_pos if reference_world_pos is None else reference_world_pos

        _, scale_origin, original_mean_pos = transform2origin(
            ref_rotated_pos, preprocessing_params["scale"]
        )

        # Transform simulated particles to MPM domain [0,2]^3 (using reference)
        transformed_pos = transform_with_reference(
            rotated_pos_sim, scale_origin, original_mean_pos
        )
        transformed_pos = shift2center111(transformed_pos)
        init_cov_sim = apply_cov_rotations(init_cov_sim, rotation_matrices)
        init_cov_sim = scale_origin * scale_origin * init_cov_sim

        # Apply per-object position_offset in MPM space (post-reference transform)
        gs_ptr = 0
        for d in obj_data:
            obj = d["obj"]
            n = d["gs_count"]
            offset = obj.get("position_offset")
            if offset is not None:
                transformed_pos[gs_ptr:gs_ptr + n] += torch.tensor(
                    offset, device="cuda", dtype=torch.float32
                )
            gs_ptr += n

        # 3e. Per-object particle filling (simulate objects only)
        gs_num = transformed_pos.shape[0]
        shared_filling = preprocessing_params["particle_filling"]

        all_filled = []
        per_object_info = []
        gs_offset = 0
        filled_offset = gs_num

        for d in obj_data:
            obj = d["obj"]
            n_gs = d["gs_count"]
            filling = obj.get("particle_filling") or shared_filling
            n_filled = 0

            if filling is not None and n_gs > 0:
                if fill_particles is None:
                    raise ModuleNotFoundError(
                        "particle_filling was requested but Taichi-based fill_particles "
                        "is unavailable. Install taichi or disable particle_filling."
                    )
                obj_pos = transformed_pos[gs_offset:gs_offset + n_gs]
                obj_opacity = init_opacity_sim[gs_offset:gs_offset + n_gs]
                obj_cov = init_cov_sim[gs_offset:gs_offset + n_gs]

                obj_mpm = fill_particles(
                    pos=obj_pos,
                    opacity=obj_opacity,
                    cov=obj_cov,
                    grid_n=filling["n_grid"],
                    max_samples=filling["max_particles_num"],
                    grid_dx=material_params["grid_lim"] / filling["n_grid"],
                    density_thres=filling["density_threshold"],
                    search_thres=filling["search_threshold"],
                    max_particles_per_cell=filling["max_partciels_per_cell"],
                    search_exclude_dir=filling["search_exclude_direction"],
                    ray_cast_dir=filling["ray_cast_direction"],
                    boundary=filling["boundary"],
                    smooth=filling["smooth"],
                ).to(device=device)

                obj_filled = obj_mpm[n_gs:]
                n_filled = obj_filled.shape[0]
                if n_filled > 0:
                    all_filled.append(obj_filled)

            gs_idx = list(range(gs_offset, gs_offset + n_gs))
            filled_idx = list(range(filled_offset, filled_offset + n_filled))
            per_object_info.append(
                dict(
                    name=obj["name"],
                    particle_indices=gs_idx + filled_idx,
                    material=obj["material"],
                )
            )
            print(
                f"    [simulate] {obj['name']}: {n_gs} gs + {n_filled} filled "
                f"= {n_gs + n_filled} total"
            )

            gs_offset += n_gs
            filled_offset += n_filled

        # Build final array: [all_gs, all_filled]
        if all_filled:
            mpm_init_pos = torch.cat([transformed_pos.to(device)] + all_filled, dim=0)
        else:
            mpm_init_pos = transformed_pos.to(device)

        mpm_init_vol = _compute_volumes(mpm_init_pos, uniform=False)

        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov_sim
        shs = init_shs_sim
        opacity = init_opacity_sim

    else:
        # ── SINGLE-OBJECT PATH (unchanged) ────────────────────────

        # 3c. Select simulation area
        if preprocessing_params["sim_area"] is not None:
            boundary = preprocessing_params["sim_area"]
            assert len(boundary) == 6
            area_mask = torch.ones(
                rotated_pos.shape[0], dtype=torch.bool, device="cuda"
            )
            for i in range(3):
                area_mask = torch.logical_and(
                    area_mask, rotated_pos[:, i] > boundary[2 * i]
                )
                area_mask = torch.logical_and(
                    area_mask, rotated_pos[:, i] < boundary[2 * i + 1]
                )

            # Keep unselected particles for rendering (world space)
            static_pos = init_pos[~area_mask, :]
            static_cov = init_cov[~area_mask, :]
            static_opacity = init_opacity[~area_mask, :]
            static_shs = init_shs[~area_mask, :]
            static_quats = init_ply_quats[~area_mask, :]
            static_scales = init_ply_scales[~area_mask, :]
            has_static = True

            rotated_pos = rotated_pos[area_mask, :]
            init_cov = init_cov[area_mask, :]
            init_opacity = init_opacity[area_mask, :]
            init_shs = init_shs[area_mask, :]
            init_ply_quats = init_ply_quats[area_mask, :]
            init_ply_scales = init_ply_scales[area_mask, :]

        if args.debug:
            _log("=== CHECKPOINT 3: After sim_area selection ===")
            _log(f"  [DBG] Selected (sim) particles: {rotated_pos.shape[0]}")
            _dbg("rotated_pos (sim only)", rotated_pos)
            if has_static:
                _log(f"  [DBG] Static render particles: {static_pos.shape[0]}")

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
            if fill_particles is None:
                raise ModuleNotFoundError(
                    "particle_filling was requested but Taichi-based fill_particles "
                    "is unavailable. Install taichi or disable particle_filling."
                )
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
        mpm_init_vol = _compute_volumes(
            mpm_init_pos, uniform=material_params["material"] == "sand"
        )

        # 3g. Initialise filled-particle attributes
        if filling_params is not None and filling_params.get("visualize", False):
            if init_filled_particles is None:
                raise ModuleNotFoundError(
                    "particle_filling.visualize was requested but Taichi-based "
                    "init_filled_particles is unavailable. Install taichi or disable visualize."
                )
            shs, opacity, mpm_init_cov = init_filled_particles(
                mpm_init_pos[:gs_num], init_shs, init_cov, init_opacity,
                mpm_init_pos[gs_num:],
            )
            gs_num = mpm_init_pos.shape[0]
        else:
            mpm_init_cov = torch.zeros(
                (mpm_init_pos.shape[0], 6), device=device
            )
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
        if per_object_info is not None:
            for info in per_object_info:
                _log(f"  [DBG] Object '{info['name']}': "
                     f"{len(info['particle_indices'])} particles")

    # ── 4. Initialise physics backend ─────────────────────────────────
    class _NoPhysicsBackend:
        """Render-only backend: keeps particles static in MPM space."""

        def __init__(self, device: str):
            self._device = device
            self._pos = None
            self._cov = None
            self._rot = None
            self._quats = None
            self._scales = None

        def initialize(self, positions, volumes, covariances, **kwargs):
            self._pos = positions.clone()
            self._cov = covariances.clone()
            n = positions.shape[0]
            self._rot = torch.eye(3, device=self._device, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
            iq = kwargs.get("init_quats")
            isc = kwargs.get("init_scales")
            self._quats = iq.clone() if iq is not None else None
            self._scales = isc.clone() if isc is not None else None

        def set_material(self, material_params: dict) -> None:
            pass

        def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
            pass

        def finalize(self) -> None:
            pass

        def step(self, dt: float, frame: int) -> None:
            pass

        def get_state(self) -> SimulationState:
            return SimulationState(
                positions=self._pos,
                covariances=self._cov,
                rotations=self._rot,
                quats=self._quats,
                scales=self._scales,
            )

    if args.backend == "none":
        print("Initialising render-only backend (no physics)...")
        backend = _NoPhysicsBackend(device=device)
    elif args.backend == "newton_mpm":
        print("Initialising physics backend (Newton-MPM)...")
        from physics_sim.backend.newton_mpm import NewtonMPMBackend

        backend = NewtonMPMBackend(device=device)
    elif args.backend == "newton_rigid":
        print("Initialising physics backend (Newton-Rigid)...")
        from physics_sim.backend.newton_rigid import NewtonRigidBackend

        backend = NewtonRigidBackend(device=device)
    elif args.backend == "newton_vbd":
        print("Initialising physics backend (Newton-VBD)...")
        from physics_sim.backend.newton_vbd import NewtonVBDBackend

        backend = NewtonVBDBackend(device=device)
    else:
        print("Initialising physics backend (Warp-MPM)...")
        from physics_sim.backend.warp_mpm import WarpMPMBackend

        backend = WarpMPMBackend(device=device)

    # Collect backend-specific kwargs for initialize()
    init_kwargs = dict(
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    # Newton rigid: pass collision geometry settings from backend overrides
    if args.backend == "newton_rigid" and args.backend in backend_overrides:
        rigid_opts = backend_overrides[args.backend]
        for k in ("collision_geometry", "alpha", "max_triangles",
                  "contact_margin", "use_sdf", "sdf_resolution",
                  "sdf_narrow_band"):
            if k in rigid_opts:
                init_kwargs[k] = rigid_opts[k]

    # Newton VBD: pass collision + tet mesh settings from backend overrides
    if args.backend == "newton_vbd" and args.backend in backend_overrides:
        vbd_opts = backend_overrides[args.backend]
        for k in ("collision_geometry", "alpha", "max_triangles",
                  "contact_margin",
                  "debug_soft_no_deformation", "sv_clamp_min",
                  "sv_clamp_max",
                  "particle_self_contact",
                  "particle_self_contact_radius",
                  "particle_self_contact_margin",
                  "soft_contact_ke", "soft_contact_kd",
                  "soft_contact_mu",
                  "rigid_contact_max"):
            if k in vbd_opts:
                init_kwargs[k] = vbd_opts[k]

    # 2DGS: preprocess quats (apply rotation chain) and pass to backend.
    # Scales are passed as-is (scale_origin cancels in the round-trip).
    if gs_type == "2dgs":
        quats_mpm = preprocess_quats(init_ply_quats, axis_perm, rotation_matrices)
        init_kwargs["init_quats"] = quats_mpm
        init_kwargs["init_scales"] = init_ply_scales

    backend.initialize(
        mpm_init_pos, mpm_init_vol, mpm_init_cov, **init_kwargs
    )
    # Pass per-object material info to backend (if multi-object)
    if per_object_info is not None:
        material_params["per_object"] = per_object_info
    backend.set_material(material_params)

    # ── 4b. Boundary conditions (optional world→MPM conversion) ────────
    # Merge user boundary_conditions with collider_only-generated planes.
    bc_all = []
    if isinstance(bc_params, list):
        bc_all.extend(bc_params)
    elif bc_params not in (None, {}, ""):
        raise TypeError(
            f"boundary_conditions must be a list (got {type(bc_params)})."
        )
    if collider_bc_params:
        bc_all.extend(collider_bc_params)

    bc_params_mpm = bc_all
    if isinstance(bc_all, list):
        bc_params_mpm = []
        for bc in bc_all:
            if bc.get("space") == "world" and bc.get("type") == "surface_collider":
                # Convert point+normal from world space to the backend's MPM space.
                p_w = torch.tensor(bc["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
                n_w = torch.tensor(bc["normal"], device="cuda", dtype=torch.float32).reshape(1, 3)
                p_m = world_to_mpm_positions(
                    p_w, rotation_matrices, scale_origin, original_mean_pos
                )[0]
                n_m = world_to_mpm_directions(n_w, rotation_matrices)[0]
                n_m = n_m / (torch.norm(n_m) + 1e-12)
                bc_new = dict(bc)
                bc_new["point"] = [float(x) for x in p_m.detach().cpu().numpy().tolist()]
                bc_new["normal"] = [float(x) for x in n_m.detach().cpu().numpy().tolist()]
                bc_new.pop("space", None)
                bc_params_mpm.append(bc_new)
            else:
                bc_params_mpm.append(bc)

    backend.set_boundary_conditions(bc_params_mpm, time_params)
    backend.finalize()  # finalize after set_boundary_conditions

    # ── 5. Optional: simulation without rendering ─────────────────────
    if args.no_render:
        print("Running simulation (no rendering)...")
        substep_dt = time_params["substep_dt"]
        frame_dt = time_params["frame_dt"]
        frame_num = time_params["frame_num"]
        step_per_frame = int(frame_dt / substep_dt)
        for frame in tqdm(range(frame_num), desc="Simulating"):
            for _ in range(step_per_frame):
                backend.step(substep_dt, frame)

        state = backend.get_state()
        pos = state.positions[:gs_num].to(device)
        cov3D = state.covariances[:gs_num].to(device)
        rot = state.rotations[:gs_num].to(device)

        # Convert back to world space for export
        pos_world = apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(pos), scale_origin, original_mean_pos
            ),
            rotation_matrices,
        )
        cov_world = cov3D / (scale_origin * scale_origin)
        cov_world = apply_inverse_cov_rotations(cov_world, rotation_matrices)

        out_path = os.path.join(args.output, "final_state.npz")
        np.savez_compressed(
            out_path,
            positions=pos_world.detach().cpu().numpy(),
            covariances=cov_world.detach().cpu().numpy(),
            rotations=rot.detach().cpu().numpy(),
        )
        print(f"Saved final state to {out_path}")
        return

    # ── 5. Camera setup ───────────────────────────────────────────────
    if args.camera_mode == "json":
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
    else:
        # Procedural cameras orbit around the reference point cloud center (world space).
        if reference_world_pos is None:
            reference_world_pos = init_pos
        lo = torch.min(reference_world_pos, dim=0)[0]
        hi = torch.max(reference_world_pos, dim=0)[0]
        viewpoint_center_worldspace = ((lo + hi) * 0.5).detach().cpu().numpy()
        # Build an observant coordinate frame from a world-up vector.
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vertical, h1, h2 = generate_local_coord(world_up)
        observant_coordinates = np.column_stack((h1, h2, vertical))

    # ── 6. Simulation + rendering loop ────────────────────────────────
    print("Running simulation and rendering...")
    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    print(f"  substep_dt={substep_dt:.2e}  frame_dt={frame_dt:.2e}  "
          f"steps/frame={step_per_frame}  frames={frame_num}")

    # Clean stale frame PNGs from previous runs to prevent ffmpeg
    # from including old frames when frame_num has been reduced.
    import glob as _glob
    stale = sorted(_glob.glob(os.path.join(args.output, "[0-9]*.png")))
    if stale:
        expected = {
            os.path.join(args.output, f"{i:04d}.png")
            for i in range(frame_num)
        }
        to_remove = [p for p in stale if p not in expected]
        if to_remove:
            print(f"  Removing {len(to_remove)} stale frame PNG(s) from previous run")
            for p in to_remove:
                os.remove(p)

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
        if args.camera_mode == "json":
            camera = renderer.build_camera_from_json(
                args.cameras_json,
                camera_params,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                current_frame=frame,
            )
        elif args.camera_mode == "orbit":
            camera = renderer.build_camera_orbit(
                camera_params=camera_params,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                current_frame=frame,
            )
        else:
            camera = renderer.build_camera_fixed(camera_params=camera_params)

        # Substep simulation
        for step in range(step_per_frame):
            backend.step(substep_dt, frame)

        # Export simulation state
        state = backend.get_state()
        pos = state.positions[:gs_num].to(device)
        cov3D = state.covariances[:gs_num].to(device)
        rot = state.rotations[:gs_num].to(device)

        # ── Per-frame rigid body diagnostics ──────────────────────────
        if hasattr(backend, "get_diagnostics"):
            diag = backend.get_diagnostics()
            for bd in diag["bodies"]:
                p = bd["pos"]
                v = bd["vel"]
                pdists = bd["plane_distances"]
                nan_flag = " *** NaN! ***" if bd["has_nan"] else ""
                dist_str = "  ".join(
                    f"plane{pi}={sd:+.4f}" for pi, sd in pdists
                )
                print(
                    f"  [DIAG] Frame {frame} {bd['name']}: "
                    f"pos=[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}] "
                    f"vel=[{v[0]:.4f},{v[1]:.4f},{v[2]:.4f}] "
                    f"{dist_str}{nan_flag}"
                )
                for pi, sd in pdists:
                    if sd < 0:
                        print(
                            f"  [DIAG] *** Frame {frame}: body PENETRATES "
                            f"plane {pi} by {-sd:.6f} ***"
                        )

        # Sanity check: detect NaN/Inf/extreme positions
        nan_mask = ~torch.isfinite(pos).all(dim=1)
        extreme_mask = (pos.abs() > 10.0).any(dim=1)
        bad_mask = nan_mask | extreme_mask
        if bad_mask.any():
            n_nan = int(nan_mask.sum().item())
            n_ext = int((extreme_mask & ~nan_mask).sum().item())
            print(
                f"[WARNING] Frame {frame}: {n_nan} NaN/Inf + {n_ext} extreme "
                f"(|pos|>10) out of {gs_num} particles"
            )
            pos[bad_mask] = 0.0
            cov3D[bad_mask] = 0.0

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

        # 2DGS: inverse-transform quats (scales need no adjustment)
        # inverse_preprocess_quats undoes both axis_perm and rotation,
        # returning to original PLY space.  We then re-apply axis_perm so
        # quats match the axis-permuted position space used for rendering.
        render_quats = None
        render_scales = None
        if gs_type == "2dgs" and state.quats is not None:
            render_quats = apply_axis_perm_to_quats(
                inverse_preprocess_quats(
                    state.quats[:gs_num].to(device), axis_perm, rotation_matrices,
                ),
                axis_perm,
            )
            render_scales = state.scales[:gs_num].to(device)

        if args.debug and frame == 0:
            _log("=== CHECKPOINT 7: Frame 0 — after inverse transform (world space) ===")
            _dbg("pos (world)", pos)
            _dbg("cov3D (world)", cov3D)

        # Merge with static (render-only) particles
        # Static quats are in original PLY space — axis_perm is applied
        # after merging so all quats end up in the same coordinate system.
        cur_opacity = opacity_render
        cur_shs = shs_render
        if has_static:
            pos = torch.cat([pos, static_pos], dim=0)
            cov3D = torch.cat([cov3D, static_cov], dim=0)
            cur_opacity = torch.cat([opacity_render, static_opacity], dim=0)
            cur_shs = torch.cat([shs_render, static_shs], dim=0)
            if render_quats is not None and static_quats is not None:
                render_quats = torch.cat([
                    render_quats,
                    apply_axis_perm_to_quats(static_quats, axis_perm),
                ], dim=0)
                render_scales = torch.cat([render_scales, static_scales], dim=0)

        if args.debug and frame == 0:
            _log("=== CHECKPOINT 8: Frame 0 — final render inputs ===")
            _dbg("pos (final)", pos)
            _dbg("cov3D (final)", cov3D)
            _dbg("cur_opacity", cur_opacity)
            _dbg("cur_shs", cur_shs)
            _log(f"  [DBG] Total render particles: {pos.shape[0]}")

        # Convert SH → RGB and rasterise
        colors_precomp = renderer.convert_sh(cur_shs, camera, pos, rot)
        if gs_type == "2dgs" and render_quats is not None:
            rendering, _meta = renderer.render(
                camera=camera,
                means=pos,
                colors=colors_precomp,
                opacities=cur_opacity,
                bg_color=bg_color,
                quats=render_quats,
                scales=render_scales,
            )
        else:
            rendering, _meta = renderer.render(
                camera=camera,
                means=pos,
                colors=colors_precomp,
                opacities=cur_opacity,
                bg_color=bg_color,
                cov6=cov3D,
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
