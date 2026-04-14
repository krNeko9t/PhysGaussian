#!/usr/bin/env python3
"""
Physics simulation + rendering pipeline for 3DGS / 2DGS scenes.

Usage:
    python pipeline.py --config experiments/wolf_bread_rigid.yaml
    python pipeline.py --config experiments/wolf_bread_rigid.yaml --white_bg --compile_video
    python pipeline.py --config experiments/wolf_bread_rigid.yaml --no_render

All scene parameters (objects, materials, backend, camera, paths) live in
the YAML config.  The config uses Hydra/OmegaConf ``defaults:`` to compose
reusable sub-configs from ``physics_sim/conf/``.
"""

import argparse
import os
import glob as _glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from physics_sim.config.loader import load_config
from physics_sim.scene import SceneObject, assemble_scene
from physics_sim.backend.base import SimulationState
from physics_sim.renderer.gs_renderer import GaussianRenderer
from physics_sim.geometry.plane_fit import fit_plane_svd
from physics_sim.preprocessing.transform import (
    generate_rotation_matrices,
    apply_rotations,
    apply_inverse_rotations,
    apply_cov_rotations,
    apply_inverse_cov_rotations,
    generate_local_coord,
    transform2origin,
    get_center_view_worldspace_and_observant_coordinate,
)


# ── Debug helpers ────────────────────────────────────────────────────────

_debug_log_path = None


def _log(msg: str):
    global _debug_log_path
    if _debug_log_path is None:
        print(msg)
    else:
        os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
        with open(_debug_log_path, "a") as f:
            f.write(msg + "\n")


def _dbg(label: str, t: torch.Tensor):
    t_f = t.float()
    _log(
        f"  [DBG] {label:40s} shape={str(list(t.shape)):20s} "
        f"mean={t_f.mean().item():12.6f}  min={t_f.min().item():12.6f}  "
        f"max={t_f.max().item():12.6f}  sum={t_f.sum().item():14.4f}"
    )


# ── Quaternion / axis-permutation helpers ────────────────────────────────
# Kept in pipeline because they're used in the render loop for 2DGS.

def _build_axis_perm_matrix(perm: str, device: torch.device) -> torch.Tensor:
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
    single = R.dim() == 2
    if single:
        R = R.unsqueeze(0)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)
    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(trace[m1] + 1.0) * 2.0
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s
    m2 = ~m1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = torch.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2.0
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s
    m3 = ~m1 & ~m2 & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = torch.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2.0
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s
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


def apply_axis_perm_to_quats(quats_wxyz: torch.Tensor, perm: str) -> torch.Tensor:
    if perm == "xyz":
        return quats_wxyz
    device = quats_wxyz.device
    P = _build_axis_perm_matrix(perm, device)
    if torch.det(P) < 0:
        P = -P
    q_perm = _rotmat_to_quat_wxyz(P)
    return _quat_mul_wxyz(q_perm, quats_wxyz)


def preprocess_quats(
    quats_wxyz: torch.Tensor,
    axis_perm: str,
    rotation_matrices: list[torch.Tensor],
) -> torch.Tensor:
    device = quats_wxyz.device
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
    device = quats_mpm_wxyz.device
    R_combined = _build_axis_perm_matrix(axis_perm, device)
    if torch.det(R_combined) < 0:
        R_combined = -R_combined
    for R in rotation_matrices:
        R_combined = R @ R_combined
    q_pre = _rotmat_to_quat_wxyz(R_combined)
    q_pre_inv = q_pre.clone()
    q_pre_inv[1:] = -q_pre_inv[1:]
    return _quat_mul_wxyz(q_pre_inv, quats_mpm_wxyz)


# ── Pipeline helpers ─────────────────────────────────────────────────────

def _estimate_volumes(pos: torch.Tensor, n_grid: int) -> torch.Tensor:
    lo = pos.min(dim=0)[0]
    hi = pos.max(dim=0)[0]
    extent = (hi - lo).max().item()
    if extent < 1e-8:
        extent = 1.0
    dx = extent / max(n_grid, 1)
    return torch.full((pos.shape[0],), float(dx ** 3), device=pos.device)


def _build_per_object_info(sim_objects: list[SceneObject]) -> list[dict]:
    """Build the ``per_object_info`` list consumed by physics backends."""
    info = []
    offset = 0
    for obj in sim_objects:
        n = obj.n_particles
        info.append(dict(
            name=obj.name,
            particle_indices=list(range(offset, offset + n)),
            material=obj.material,
        ))
        offset += n
    return info


def _resolve_collider_bc(
    collider_objects: list[SceneObject],
    renderer: GaussianRenderer,
    axis_perm: str,
    opacity_threshold: float,
) -> list[dict]:
    """Convert collider_only objects into surface_collider boundary conditions."""
    bc_list: list[dict] = []
    for obj in collider_objects:
        col = obj.collider or {}
        col_type = col.get("type", "plane")
        if col_type != "plane":
            raise ValueError(
                f"collider_only object '{obj.name}' only supports "
                f"collider.type='plane' for now (got {col_type!r})"
            )

        space = col.get("space", "world")
        surface = col.get("surface", "sticky")
        friction = float(col.get("friction", 0.0))
        start_time = col.get("start_time", 0)
        end_time = col.get("end_time", 1e3)

        if col.get("point") is not None and col.get("normal") is not None:
            point = col["point"]
            normal = col["normal"]
        else:
            pts = obj.positions.detach().cpu().numpy()
            if pts.shape[0] < 3:
                raise ValueError(
                    f"collider_only object '{obj.name}' has too few points "
                    "for plane fitting and no explicit point+normal."
                )
            fit = col.get("fit") or {}
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
                f"    [collider] {obj.name}: plane rms={res.rms:.6f}, "
                f"point={point}, normal={normal}"
            )

        bc_list.append(dict(
            type="surface_collider",
            space=space,
            point=point,
            normal=normal,
            surface=surface,
            friction=friction,
            start_time=start_time,
            end_time=end_time,
        ))
    return bc_list


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3DGS Physics Simulation Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a Hydra-composable YAML config")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--no_render", action="store_true",
                        help="Run physics only, skip rendering.")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree of PLY models (default: 3)")
    parser.add_argument("--raster_backend", type=str, default="gsplat",
                        choices=["gsplat", "diffrast"],
                        help="Rasterization backend (default: gsplat)")
    parser.add_argument("--debug", action="store_true",
                        help="Print intermediate tensor statistics")
    parser.add_argument("--debug_log", type=str, default=None,
                        help="Path to debug log file")
    args = parser.parse_args()

    assert os.path.exists(args.config), f"Config not found: {args.config}"

    # ── 1. Load config ───────────────────────────────────────────────
    print("Loading config...")
    cfg = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    backend_name = cfg.get("backend", "warp_mpm")
    output_dir = cfg.get("output", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Debug setup
    if args.debug:
        global _debug_log_path
        if args.debug_log is not None:
            _debug_log_path = args.debug_log
        else:
            _debug_log_path = os.path.join(output_dir, "debug.log")
        os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
        with open(_debug_log_path, "w"):
            pass

    device = "cuda:0"

    # ── 2. Initialise runtime ────────────────────────────────────────
    if backend_name != "none":
        try:
            import warp as wp  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Python module 'warp' not found. Install warp_lang."
            ) from e
        try:
            import taichi as ti  # type: ignore
        except ModuleNotFoundError:
            ti = None

        wp.init()
        if backend_name in ("newton_mpm", "newton_rigid", "newton_vbd"):
            wp.config.verify_cuda = False
        else:
            wp.config.verify_cuda = True

        if ti is not None:
            ti.init(arch=ti.cuda, device_memory_GB=8.0)

    # ── 3. Assemble scene objects ────────────────────────────────────
    print("Assembling scene...")
    raster_be = "gsplat" if args.no_render else args.raster_backend
    renderer = GaussianRenderer(sh_degree=args.sh_degree, raster_backend=raster_be)

    objects = assemble_scene(cfg, renderer, config_dir=config_dir)

    sim_objects = [o for o in objects if o.mode == "simulate"]
    static_objects = [o for o in objects if o.mode == "render_only"]
    collider_objects = [o for o in objects if o.mode == "collider_only"]

    if not sim_objects and backend_name != "none":
        raise ValueError("No simulate objects found — nothing to simulate.")

    # Determine gs_type from the first simulate object (or first overall).
    gs_type = (sim_objects or objects)[0].gs_type

    # Preprocess config
    preprocess = cfg.get("preprocess", {})
    axis_perm = preprocess.get("axis_permutation", "xyz")
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocess.get("rotation_degree", [0.0])),
        preprocess.get("rotation_axis", [0]),
    )

    # ── 4. Concatenate simulation particles ──────────────────────────
    n_grid = cfg.get("n_grid", 200)
    grid_lim = cfg.get("grid_lim", 2.0)

    if sim_objects:
        sim_init_pos = torch.cat([o.positions for o in sim_objects], dim=0).to(device)
        sim_init_cov = torch.cat([o.covariances for o in sim_objects], dim=0).to(device)
        sim_init_vol = _estimate_volumes(sim_init_pos, n_grid)
        sim_shs = torch.cat([o.shs for o in sim_objects], dim=0)
        sim_opacity = torch.cat([o.opacities for o in sim_objects], dim=0)
        sim_quats = torch.cat([o.quats for o in sim_objects], dim=0)
        sim_scales = torch.cat([o.scales for o in sim_objects], dim=0)
        gs_num = sim_init_pos.shape[0]
        per_object_info = _build_per_object_info(sim_objects)
    else:
        # No simulation — create empty tensors for rendering path
        sim_init_pos = torch.zeros(0, 3, device=device)
        sim_init_cov = torch.zeros(0, 6, device=device)
        sim_init_vol = torch.zeros(0, device=device)
        sim_shs = torch.zeros(0, (args.sh_degree + 1) ** 2, 3, device=device)
        sim_opacity = torch.zeros(0, 1, device=device)
        sim_quats = torch.zeros(0, 4, device=device)
        sim_scales = torch.zeros(0, 3, device=device)
        gs_num = 0
        per_object_info = []

    # Build static render data from render_only + collider_only (with render flag)
    static_chunks = []
    for obj in static_objects:
        static_chunks.append(obj)
    for obj in collider_objects:
        render_flag = (obj.collider or {}).get("render", True)
        if render_flag and obj.n_particles > 0:
            static_chunks.append(obj)

    has_static = len(static_chunks) > 0
    if has_static:
        static_pos = torch.cat([o.positions for o in static_chunks], dim=0)
        static_cov = torch.cat([o.covariances for o in static_chunks], dim=0)
        static_opacity = torch.cat([o.opacities for o in static_chunks], dim=0)
        static_shs = torch.cat([o.shs for o in static_chunks], dim=0)
        static_quats = torch.cat([o.quats for o in static_chunks], dim=0)
        static_scales = torch.cat([o.scales for o in static_chunks], dim=0)
    else:
        static_pos = static_cov = static_opacity = static_shs = None
        static_quats = static_scales = None

    if args.debug:
        _log(f"=== Scene assembled: {gs_num} sim particles, "
             f"{sum(o.n_particles for o in static_chunks) if has_static else 0} static ===")
        if gs_num > 0:
            _dbg("sim_init_pos", sim_init_pos)
            _dbg("sim_init_cov", sim_init_cov)
        for info in per_object_info:
            _log(f"  [DBG] Object '{info['name']}': {len(info['particle_indices'])} particles")

    # ── 5. Initialise physics backend ────────────────────────────────
    class _NoPhysicsBackend:
        def __init__(self, dev: str):
            self._device = dev
            self._pos = self._cov = self._rot = None
            self._quats = self._scales = None

        def initialize(self, positions, volumes, covariances, **kwargs):
            self._pos = positions.clone()
            self._cov = covariances.clone()
            n = positions.shape[0]
            self._rot = torch.eye(3, device=self._device, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
            self._quats = kwargs.get("init_quats")
            self._scales = kwargs.get("init_scales")
            if self._quats is not None:
                self._quats = self._quats.clone()
            if self._scales is not None:
                self._scales = self._scales.clone()

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
                positions=self._pos, covariances=self._cov,
                rotations=self._rot, quats=self._quats, scales=self._scales,
            )

    print(f"Initialising backend: {backend_name}")
    if backend_name == "none":
        backend = _NoPhysicsBackend(device)
    elif backend_name == "newton_mpm":
        from physics_sim.backend.newton_mpm import NewtonMPMBackend
        backend = NewtonMPMBackend(device=device)
    elif backend_name == "newton_rigid":
        from physics_sim.backend.newton_rigid import NewtonRigidBackend
        backend = NewtonRigidBackend(device=device)
    elif backend_name == "newton_vbd":
        from physics_sim.backend.newton_vbd import NewtonVBDBackend
        backend = NewtonVBDBackend(device=device)
    else:
        from physics_sim.backend.warp_mpm import WarpMPMBackend
        backend = WarpMPMBackend(device=device)

    # Collect init kwargs from backend config
    backend_cfg = cfg.get("backend_opts", {})
    init_kwargs: dict = dict(n_grid=n_grid, grid_lim=grid_lim)

    _INIT_PASSTHROUGH_KEYS = (
        "collision_geometry", "alpha", "max_triangles",
        "contact_margin", "use_sdf", "sdf_resolution", "sdf_narrow_band",
        "debug_soft_no_deformation", "sv_clamp_min", "sv_clamp_max",
        "particle_self_contact", "particle_self_contact_radius",
        "particle_self_contact_margin",
        "soft_contact_ke", "soft_contact_kd", "soft_contact_mu",
        "rigid_contact_max",
    )
    for k in _INIT_PASSTHROUGH_KEYS:
        val = cfg.get(k)
        if val is None:
            val = backend_cfg.get(k)
        if val is not None:
            init_kwargs[k] = val

    # 2DGS: preprocess quats
    if gs_type == "2dgs" and gs_num > 0:
        quats_sim = preprocess_quats(sim_quats, axis_perm, rotation_matrices)
        init_kwargs["init_quats"] = quats_sim
        init_kwargs["init_scales"] = sim_scales

    # Warp-MPM extras
    if backend_name == "warp_mpm":
        init_kwargs["scale"] = preprocess.get("scale", 1.0)
        init_kwargs["opacity"] = sim_opacity
        filling = preprocess.get("particle_filling")
        if filling is not None:
            init_kwargs["filling_params"] = filling

    if gs_num > 0:
        backend.initialize(sim_init_pos, sim_init_vol, sim_init_cov, **init_kwargs)

    # Material params: build top-level + per-object
    material_params: dict = {}
    for key in ("material", "E", "nu", "density", "g", "friction_angle",
                "yield_stress", "hardening", "xi", "plastic_viscosity",
                "softening", "rpic_damping", "pic_damping",
                "ke", "kd", "mu", "friction",
                "n_grid", "grid_lim", "grid_v_damping_scale"):
        if key in cfg:
            material_params[key] = cfg[key]
    material_params.setdefault("n_grid", n_grid)
    material_params.setdefault("grid_lim", grid_lim)

    # Solver options
    solver_opts = cfg.get("solver")
    if solver_opts is not None:
        material_params["newton_solver_opts"] = solver_opts

    if per_object_info:
        material_params["per_object"] = per_object_info
    backend.set_material(material_params)

    # ── 6. Boundary conditions ───────────────────────────────────────
    bc_all: list[dict] = list(cfg.get("boundary_conditions", []))

    # Add collider-only generated BCs
    collider_bcs = _resolve_collider_bc(
        collider_objects, renderer, axis_perm,
        preprocess.get("opacity_threshold", 0.02),
    )
    bc_all.extend(collider_bcs)

    # Rotate world-space BCs into backend space
    bc_converted: list[dict] = []
    for bc in bc_all:
        if bc.get("space") == "world" and bc.get("type") == "surface_collider":
            p_w = torch.tensor(bc["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
            n_w = torch.tensor(bc["normal"], device="cuda", dtype=torch.float32).reshape(1, 3)
            p_r = apply_rotations(p_w, rotation_matrices)[0]
            n_r = apply_rotations(n_w, rotation_matrices)[0]
            n_r = n_r / (torch.norm(n_r) + 1e-12)
            bc_new = dict(bc)
            bc_new["point"] = [float(x) for x in p_r.detach().cpu().tolist()]
            bc_new["normal"] = [float(x) for x in n_r.detach().cpu().tolist()]
            bc_new.pop("space", None)
            bc_converted.append(bc_new)
        else:
            bc_converted.append(bc)

    time_params = {
        "substep_dt": cfg.get("substep_dt", 1e-4),
        "frame_dt": cfg.get("frame_dt", 1e-2),
        "frame_num": cfg.get("frame_num", 100),
    }

    backend.set_boundary_conditions(bc_converted, time_params)
    backend.finalize()

    # ── 7. No-render path ────────────────────────────────────────────
    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)

    if args.no_render:
        print("Running simulation (no rendering)...")
        for frame in tqdm(range(frame_num), desc="Simulating"):
            for _ in range(step_per_frame):
                backend.step(substep_dt, frame)

        state = backend.get_state()
        pos = state.positions[:gs_num].to(device)
        cov3D = state.covariances[:gs_num].to(device)
        rot = state.rotations[:gs_num].to(device)

        pos_world = apply_inverse_rotations(pos, rotation_matrices)
        cov_world = apply_inverse_cov_rotations(cov3D, rotation_matrices)

        out_path = os.path.join(output_dir, "final_state.npz")
        np.savez_compressed(
            out_path,
            positions=pos_world.detach().cpu().numpy(),
            covariances=cov_world.detach().cpu().numpy(),
            rotations=rot.detach().cpu().numpy(),
        )
        print(f"Saved final state to {out_path}")
        return

    # ── 8. Camera setup ──────────────────────────────────────────────
    camera_cfg = cfg.get("camera", cfg)
    camera_mode = camera_cfg.get("camera_mode", "orbit")

    # Resolve cameras_json path for json mode
    cameras_json = camera_cfg.get("cameras_json")

    if camera_mode == "json":
        if cameras_json is None:
            raise FileNotFoundError(
                "camera_mode='json' requires 'cameras_json' in the config."
            )
        cameras_json = str(cameras_json)
        if not os.path.isabs(cameras_json):
            cameras_json = os.path.join(config_dir, cameras_json)
        assert os.path.exists(cameras_json), f"cameras.json not found: {cameras_json}"

        mpm_space_viewpoint_center = torch.tensor(
            camera_cfg.get("mpm_space_viewpoint_center", [1.0, 1.0, 1.0])
        ).reshape(1, 3).cuda()
        mpm_space_vertical_upward_axis = torch.tensor(
            camera_cfg.get("mpm_space_vertical_upward_axis", [0, 0, 1])
        ).reshape(1, 3).cuda()

        _cam_ref_pos = sim_init_pos
        _, _cam_scale_origin, _cam_mean_pos = transform2origin(
            _cam_ref_pos, preprocess.get("scale", 1.0),
        )
        viewpoint_center_worldspace, observant_coordinates = (
            get_center_view_worldspace_and_observant_coordinate(
                mpm_space_viewpoint_center, mpm_space_vertical_upward_axis,
                rotation_matrices, _cam_scale_origin, _cam_mean_pos,
            )
        )
    else:
        # Procedural cameras: orbit or fixed
        if sim_objects:
            ref_pos = sim_init_pos
        elif has_static:
            ref_pos = static_pos
        else:
            ref_pos = torch.zeros(1, 3, device="cuda")
        lo = torch.min(ref_pos, dim=0)[0]
        hi = torch.max(ref_pos, dim=0)[0]
        viewpoint_center_worldspace = ((lo + hi) * 0.5).detach().cpu().numpy()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vertical, h1, h2 = generate_local_coord(world_up)
        observant_coordinates = np.column_stack((h1, h2, vertical))

    # Camera params dict (for renderer.build_camera_* methods)
    camera_params = {}
    for k in ("mpm_space_viewpoint_center", "mpm_space_vertical_upward_axis",
              "default_camera_index", "show_hint",
              "init_azimuth", "init_azimuthm", "init_elevation", "init_radius",
              "delta_a", "delta_e", "delta_r", "move_camera",
              "width", "height", "fx", "fy", "fovx_deg", "fovy_deg",
              "fixed_position", "fixed_rotation"):
        val = camera_cfg.get(k)
        if val is not None:
            camera_params[k] = val

    # ── 9. Simulation + rendering loop ───────────────────────────────
    print("Running simulation and rendering...")
    print(f"  substep_dt={substep_dt:.2e}  frame_dt={frame_dt:.2e}  "
          f"steps/frame={step_per_frame}  frames={frame_num}")

    # Clean stale frame PNGs
    stale = sorted(_glob.glob(os.path.join(output_dir, "[0-9]*.png")))
    if stale:
        expected = {os.path.join(output_dir, f"{i:04d}.png") for i in range(frame_num)}
        to_remove = [p for p in stale if p not in expected]
        if to_remove:
            print(f"  Removing {len(to_remove)} stale frame PNG(s)")
            for p in to_remove:
                os.remove(p)

    bg_color = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )
    opacity_render = sim_opacity
    shs_render = sim_shs
    height = width = None

    for frame in tqdm(range(frame_num), desc="Simulating"):
        # Build camera
        if camera_mode == "json":
            camera = renderer.build_camera_from_json(
                cameras_json, camera_params,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                current_frame=frame,
            )
        elif camera_mode == "orbit":
            camera = renderer.build_camera_orbit(
                camera_params=camera_params,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                current_frame=frame,
            )
        else:
            camera = renderer.build_camera_fixed(camera_params=camera_params)

        # Substep simulation
        for _ in range(step_per_frame):
            backend.step(substep_dt, frame)

        # Export state
        state = backend.get_state()
        pos = state.positions[:gs_num].to(device)
        cov3D = state.covariances[:gs_num].to(device)
        rot = state.rotations[:gs_num].to(device)

        # Per-frame diagnostics
        if hasattr(backend, "get_diagnostics"):
            diag = backend.get_diagnostics()
            for bd in diag["bodies"]:
                p, v = bd["pos"], bd["vel"]
                pdists = bd["plane_distances"]
                nan_flag = " *** NaN! ***" if bd["has_nan"] else ""
                dist_str = "  ".join(f"plane{pi}={sd:+.4f}" for pi, sd in pdists)
                print(
                    f"  [DIAG] Frame {frame} {bd['name']}: "
                    f"pos=[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}] "
                    f"vel=[{v[0]:.4f},{v[1]:.4f},{v[2]:.4f}] "
                    f"{dist_str}{nan_flag}"
                )

        # Sanity check
        nan_mask = ~torch.isfinite(pos).all(dim=1)
        extreme_mask = (pos.abs() > 100.0).any(dim=1)
        bad_mask = nan_mask | extreme_mask
        if bad_mask.any():
            n_nan = int(nan_mask.sum().item())
            n_ext = int((extreme_mask & ~nan_mask).sum().item())
            print(f"[WARNING] Frame {frame}: {n_nan} NaN/Inf + {n_ext} extreme particles")
            pos[bad_mask] = 0.0
            cov3D[bad_mask] = 0.0

        # Inverse rotation back to world space
        pos = apply_inverse_rotations(pos, rotation_matrices)
        cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)

        # 2DGS quats
        render_quats = render_scales = None
        if gs_type == "2dgs" and state.quats is not None:
            render_quats = apply_axis_perm_to_quats(
                inverse_preprocess_quats(
                    state.quats[:gs_num].to(device), axis_perm, rotation_matrices,
                ),
                axis_perm,
            )
            render_scales = state.scales[:gs_num].to(device)

        # Merge with static particles
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

        # Render
        colors_precomp = renderer.convert_sh(cur_shs, camera, pos, rot)
        if gs_type == "2dgs" and render_quats is not None:
            rendering, _meta = renderer.render(
                camera=camera, means=pos, colors=colors_precomp,
                opacities=cur_opacity, bg_color=bg_color,
                quats=render_quats, scales=render_scales,
            )
        else:
            rendering, _meta = renderer.render(
                camera=camera, means=pos, colors=colors_precomp,
                opacities=cur_opacity, bg_color=bg_color, cov6=cov3D,
            )

        # Save frame
        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        if height is None or width is None:
            height = cv2_img.shape[0] // 2 * 2
            width = cv2_img.shape[1] // 2 * 2
        cv2.imwrite(
            os.path.join(output_dir, f"{frame:04d}.png"),
            255 * cv2_img,
        )

    # ── 10. Compile video ────────────────────────────────────────────
    if args.compile_video:
        fps = int(1.0 / frame_dt)
        cmd = (
            f"ffmpeg -framerate {fps} -i {output_dir}/%04d.png "
            f"-c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p "
            f"{output_dir}/output.mp4"
        )
        print(f"Compiling video: {cmd}")
        os.system(cmd)
        print(f"Video saved to {output_dir}/output.mp4")

    print("Done!")


if __name__ == "__main__":
    main()
