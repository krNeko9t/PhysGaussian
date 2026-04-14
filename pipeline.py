#!/usr/bin/env python3
"""
Physics simulation + rendering pipeline for 3DGS / 2DGS scenes.

Usage:
    python pipeline.py --config experiments/wolf_bread_rigid.yaml
    python pipeline.py --config experiments/wolf_bread_rigid.yaml --white_bg --compile_video
    python pipeline.py --config experiments/wolf_bread_rigid.yaml --no_render

All scene parameters live in a YAML config file.  Config sections
(backend, material, time, preprocess, camera) can reference reusable
fragments via the ``_ref`` mechanism — see ``physics_sim/config/loader.py``.
"""

from __future__ import annotations

import argparse
import os
import glob as _glob
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from physics_sim.config.loader import ConfigLoader
from physics_sim.config.schema import SimConfig
from physics_sim.scene import SceneObject, assemble_scene
from physics_sim.backend.base import SimulationState
from physics_sim.renderer.gs_renderer import GaussianRenderer
from physics_sim.geometry.plane_fit import fit_plane_svd
from physics_sim.preprocessing.quaternions import (
    apply_axis_perm_to_quats,
    preprocess_quats,
    inverse_preprocess_quats,
)
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

_debug_log_path: Optional[str] = None


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


# ── No-op backend (backend_type == "none") ───────────────────────────────

class _NoPhysicsBackend:
    def __init__(self, dev: str):
        self._device = dev
        self._pos = self._cov = self._rot = None
        self._quats = self._scales = None

    def initialize(self, positions, volumes, covariances, **kwargs):
        self._pos = positions.clone()
        self._cov = covariances.clone()
        n = positions.shape[0]
        self._rot = (
            torch.eye(3, device=self._device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(n, 1, 1)
        )
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
            positions=self._pos,
            covariances=self._cov,
            rotations=self._rot,
            quats=self._quats,
            scales=self._scales,
        )


# ── CLI args (separate from config) ─────────────────────────────────────

@dataclass
class PipelineArgs:
    """Command-line flags that are orthogonal to the YAML config."""

    config: str = ""
    no_render: bool = False
    white_bg: bool = False
    compile_video: bool = False
    debug: bool = False
    debug_log: Optional[str] = None
    sh_degree: int = 3
    raster_backend: str = "gsplat"


# ── SimulationPipeline ───────────────────────────────────────────────────

class SimulationPipeline:
    """Orchestrates config loading, scene assembly, physics, and rendering."""

    def __init__(self, cfg: SimConfig, args: PipelineArgs):
        self.cfg = cfg
        self.args = args
        self.device = "cuda:0"
        self.config_dir = ""

        # Populated by _assemble_scene
        self.renderer: GaussianRenderer = None  # type: ignore[assignment]
        self.sim_objects: list[SceneObject] = []
        self.static_chunks: list[SceneObject] = []
        self.collider_objects: list[SceneObject] = []
        self.gs_type: str = "3dgs"
        self.gs_num: int = 0
        self.per_object_info: list[dict] = []

        # Simulation tensors (populated by _assemble_scene)
        self.sim_init_pos: torch.Tensor = torch.zeros(0, 3)
        self.sim_init_cov: torch.Tensor = torch.zeros(0, 6)
        self.sim_init_vol: torch.Tensor = torch.zeros(0)
        self.sim_shs: torch.Tensor = torch.zeros(0)
        self.sim_opacity: torch.Tensor = torch.zeros(0)
        self.sim_quats: torch.Tensor = torch.zeros(0, 4)
        self.sim_scales: torch.Tensor = torch.zeros(0, 3)

        # Static tensors
        self.static_pos: Optional[torch.Tensor] = None
        self.static_cov: Optional[torch.Tensor] = None
        self.static_opacity: Optional[torch.Tensor] = None
        self.static_shs: Optional[torch.Tensor] = None
        self.static_quats: Optional[torch.Tensor] = None
        self.static_scales: Optional[torch.Tensor] = None

        # Preprocessing state
        self.axis_perm: str = "xyz"
        self.rotation_matrices: list[torch.Tensor] = []

        # Physics backend
        self.backend = None

        # Camera state (populated by _setup_camera)
        self.viewpoint_center_worldspace = None
        self.observant_coordinates = None
        self.camera_params: dict = {}

    def run(self, config_dir: str = ""):
        self.config_dir = config_dir
        os.makedirs(self.cfg.output, exist_ok=True)
        self._setup_debug()
        self._init_runtime()
        self._assemble_scene()
        self._init_backend()
        self._setup_boundary_conditions()
        if self.args.no_render:
            self._run_headless()
        else:
            self._setup_camera()
            self._run_with_rendering()

    # ── Stage 0: Debug setup ─────────────────────────────────────────

    def _setup_debug(self):
        if not self.args.debug:
            return
        global _debug_log_path
        if self.args.debug_log is not None:
            _debug_log_path = self.args.debug_log
        else:
            _debug_log_path = os.path.join(self.cfg.output, "debug.log")
        os.makedirs(os.path.dirname(_debug_log_path), exist_ok=True)
        with open(_debug_log_path, "w"):
            pass

    # ── Stage 1: Runtime init ────────────────────────────────────────

    def _init_runtime(self):
        if self.cfg.backend_type == "none":
            return
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
        if self.cfg.backend_type in ("newton_mpm", "newton_rigid", "newton_vbd"):
            wp.config.verify_cuda = False
        else:
            wp.config.verify_cuda = True

        if ti is not None:
            ti.init(arch=ti.cuda, device_memory_GB=8.0)

    # ── Stage 2: Scene assembly ──────────────────────────────────────

    def _assemble_scene(self):
        print("Assembling scene...")
        raster_be = (
            "gsplat" if self.args.no_render else self.args.raster_backend
        )
        self.renderer = GaussianRenderer(
            sh_degree=self.args.sh_degree, raster_backend=raster_be,
        )

        objects = assemble_scene(
            self.cfg, self.renderer, config_dir=self.config_dir,
        )

        self.sim_objects = [o for o in objects if o.mode == "simulate"]
        static_objects = [o for o in objects if o.mode == "render_only"]
        self.collider_objects = [o for o in objects if o.mode == "collider_only"]

        if not self.sim_objects and self.cfg.backend_type != "none":
            raise ValueError("No simulate objects found — nothing to simulate.")

        self.gs_type = (self.sim_objects or objects)[0].gs_type

        pp = self.cfg.preprocess
        self.axis_perm = pp.axis_permutation
        self.rotation_matrices = generate_rotation_matrices(
            torch.tensor(pp.rotation_degree), pp.rotation_axis,
        )

        n_grid = self.cfg.backend.get("n_grid", 200)
        device = self.device

        if self.sim_objects:
            self.sim_init_pos = torch.cat(
                [o.positions for o in self.sim_objects], dim=0
            ).to(device)
            self.sim_init_cov = torch.cat(
                [o.covariances for o in self.sim_objects], dim=0
            ).to(device)
            self.sim_init_vol = _estimate_volumes(self.sim_init_pos, n_grid)
            self.sim_shs = torch.cat(
                [o.shs for o in self.sim_objects], dim=0
            )
            self.sim_opacity = torch.cat(
                [o.opacities for o in self.sim_objects], dim=0
            )
            self.sim_quats = torch.cat(
                [o.quats for o in self.sim_objects], dim=0
            )
            self.sim_scales = torch.cat(
                [o.scales for o in self.sim_objects], dim=0
            )
            self.gs_num = self.sim_init_pos.shape[0]
            self.per_object_info = _build_per_object_info(self.sim_objects)
        else:
            sh_c = (self.args.sh_degree + 1) ** 2
            self.sim_init_pos = torch.zeros(0, 3, device=device)
            self.sim_init_cov = torch.zeros(0, 6, device=device)
            self.sim_init_vol = torch.zeros(0, device=device)
            self.sim_shs = torch.zeros(0, sh_c, 3, device=device)
            self.sim_opacity = torch.zeros(0, 1, device=device)
            self.sim_quats = torch.zeros(0, 4, device=device)
            self.sim_scales = torch.zeros(0, 3, device=device)
            self.gs_num = 0
            self.per_object_info = []

        self.static_chunks = list(static_objects)
        for obj in self.collider_objects:
            render_flag = (obj.collider or {}).get("render", True)
            if render_flag and obj.n_particles > 0:
                self.static_chunks.append(obj)

        if self.static_chunks:
            self.static_pos = torch.cat(
                [o.positions for o in self.static_chunks], dim=0
            )
            self.static_cov = torch.cat(
                [o.covariances for o in self.static_chunks], dim=0
            )
            self.static_opacity = torch.cat(
                [o.opacities for o in self.static_chunks], dim=0
            )
            self.static_shs = torch.cat(
                [o.shs for o in self.static_chunks], dim=0
            )
            self.static_quats = torch.cat(
                [o.quats for o in self.static_chunks], dim=0
            )
            self.static_scales = torch.cat(
                [o.scales for o in self.static_chunks], dim=0
            )

        if self.args.debug:
            n_static = (
                sum(o.n_particles for o in self.static_chunks)
                if self.static_chunks
                else 0
            )
            _log(
                f"=== Scene assembled: {self.gs_num} sim particles, "
                f"{n_static} static ==="
            )
            if self.gs_num > 0:
                _dbg("sim_init_pos", self.sim_init_pos)
                _dbg("sim_init_cov", self.sim_init_cov)
            for info in self.per_object_info:
                _log(
                    f"  [DBG] Object '{info['name']}': "
                    f"{len(info['particle_indices'])} particles"
                )

    # ── Stage 3: Backend init ────────────────────────────────────────

    def _init_backend(self):
        bt = self.cfg.backend_type
        device = self.device
        print(f"Initialising backend: {bt}")

        if bt == "none":
            self.backend = _NoPhysicsBackend(device)
        elif bt == "newton_mpm":
            from physics_sim.backend.newton_mpm import NewtonMPMBackend
            self.backend = NewtonMPMBackend(device=device)
        elif bt == "newton_rigid":
            from physics_sim.backend.newton_rigid import NewtonRigidBackend
            self.backend = NewtonRigidBackend(device=device)
        elif bt == "newton_vbd":
            from physics_sim.backend.newton_vbd import NewtonVBDBackend
            self.backend = NewtonVBDBackend(device=device)
        else:
            from physics_sim.backend.warp_mpm import WarpMPMBackend
            self.backend = WarpMPMBackend(device=device)

        init_kwargs = dict(self.cfg.backend)

        if self.gs_type == "2dgs" and self.gs_num > 0:
            init_kwargs["init_quats"] = preprocess_quats(
                self.sim_quats, self.axis_perm, self.rotation_matrices,
            )
            init_kwargs["init_scales"] = self.sim_scales

        if bt == "warp_mpm":
            init_kwargs["scale"] = self.cfg.preprocess.scale
            init_kwargs["opacity"] = self.sim_opacity
            if self.cfg.particle_filling is not None:
                init_kwargs["filling_params"] = self.cfg.particle_filling

        if self.gs_num > 0:
            self.backend.initialize(
                self.sim_init_pos, self.sim_init_vol, self.sim_init_cov,
                **init_kwargs,
            )

        material_params = dict(self.cfg.material)
        material_params.setdefault("n_grid", self.cfg.backend.get("n_grid", 200))
        material_params.setdefault("grid_lim", self.cfg.backend.get("grid_lim", 2.0))

        solver_opts = self.cfg.backend.get("solver")
        if solver_opts is not None:
            material_params["newton_solver_opts"] = solver_opts

        if self.per_object_info:
            material_params["per_object"] = self.per_object_info
        self.backend.set_material(material_params)

    # ── Stage 4: Boundary conditions ─────────────────────────────────

    def _setup_boundary_conditions(self):
        bc_all: list[dict] = list(self.cfg.boundary_conditions)

        collider_bcs = _resolve_collider_bc(
            self.collider_objects,
            self.renderer,
            self.axis_perm,
            self.cfg.preprocess.opacity_threshold,
        )
        bc_all.extend(collider_bcs)

        bc_converted: list[dict] = []
        for bc in bc_all:
            if (
                bc.get("space") == "world"
                and bc.get("type") == "surface_collider"
            ):
                p_w = torch.tensor(
                    bc["point"], device="cuda", dtype=torch.float32
                ).reshape(1, 3)
                n_w = torch.tensor(
                    bc["normal"], device="cuda", dtype=torch.float32
                ).reshape(1, 3)
                p_r = apply_rotations(p_w, self.rotation_matrices)[0]
                n_r = apply_rotations(n_w, self.rotation_matrices)[0]
                n_r = n_r / (torch.norm(n_r) + 1e-12)
                bc_new = dict(bc)
                bc_new["point"] = [float(x) for x in p_r.detach().cpu().tolist()]
                bc_new["normal"] = [float(x) for x in n_r.detach().cpu().tolist()]
                bc_new.pop("space", None)
                bc_converted.append(bc_new)
            else:
                bc_converted.append(bc)

        tc = self.cfg.time
        time_params = {
            "substep_dt": tc.substep_dt,
            "frame_dt": tc.frame_dt,
            "frame_num": tc.frame_num,
        }

        self.backend.set_boundary_conditions(bc_converted, time_params)
        self.backend.finalize()

    # ── Stage 5: Camera setup ────────────────────────────────────────

    def _setup_camera(self):
        cam = self.cfg.camera
        camera_mode = cam.camera_mode

        cameras_json = cam.cameras_json
        if camera_mode == "json":
            if cameras_json is None:
                raise FileNotFoundError(
                    "camera_mode='json' requires 'cameras_json' in the config."
                )
            cameras_json = str(cameras_json)
            if not os.path.isabs(cameras_json):
                cameras_json = os.path.join(self.config_dir, cameras_json)
            assert os.path.exists(cameras_json), (
                f"cameras.json not found: {cameras_json}"
            )

            mpm_vc = torch.tensor(
                cam.mpm_space_viewpoint_center
            ).reshape(1, 3).cuda()
            mpm_up = torch.tensor(
                cam.mpm_space_vertical_upward_axis
            ).reshape(1, 3).cuda()

            _, scale_origin, mean_pos = transform2origin(
                self.sim_init_pos, self.cfg.preprocess.scale,
            )
            self.viewpoint_center_worldspace, self.observant_coordinates = (
                get_center_view_worldspace_and_observant_coordinate(
                    mpm_vc, mpm_up,
                    self.rotation_matrices, scale_origin, mean_pos,
                )
            )
        else:
            if self.sim_objects:
                ref_pos = self.sim_init_pos
            elif self.static_chunks:
                ref_pos = self.static_pos
            else:
                ref_pos = torch.zeros(1, 3, device="cuda")
            lo = torch.min(ref_pos, dim=0)[0]
            hi = torch.max(ref_pos, dim=0)[0]
            self.viewpoint_center_worldspace = (
                ((lo + hi) * 0.5).detach().cpu().numpy()
            )
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            vertical, h1, h2 = generate_local_coord(world_up)
            self.observant_coordinates = np.column_stack((h1, h2, vertical))

        cam_dict = asdict(cam)
        self.camera_params = {
            k: v for k, v in cam_dict.items() if v is not None
        }

    # ── Stage 6a: Headless simulation ────────────────────────────────

    def _run_headless(self):
        tc = self.cfg.time
        substep_dt = tc.substep_dt
        step_per_frame = int(tc.frame_dt / substep_dt)

        print("Running simulation (no rendering)...")
        for frame in tqdm(range(tc.frame_num), desc="Simulating"):
            for _ in range(step_per_frame):
                self.backend.step(substep_dt, frame)

        state = self.backend.get_state()
        pos = state.positions[: self.gs_num].to(self.device)
        cov3D = state.covariances[: self.gs_num].to(self.device)
        rot = state.rotations[: self.gs_num].to(self.device)

        pos_world = apply_inverse_rotations(pos, self.rotation_matrices)
        cov_world = apply_inverse_cov_rotations(cov3D, self.rotation_matrices)

        out_path = os.path.join(self.cfg.output, "final_state.npz")
        np.savez_compressed(
            out_path,
            positions=pos_world.detach().cpu().numpy(),
            covariances=cov_world.detach().cpu().numpy(),
            rotations=rot.detach().cpu().numpy(),
        )
        print(f"Saved final state to {out_path}")

    # ── Stage 6b: Simulation + rendering ─────────────────────────────

    def _run_with_rendering(self):
        tc = self.cfg.time
        substep_dt = tc.substep_dt
        frame_dt = tc.frame_dt
        frame_num = tc.frame_num
        step_per_frame = int(frame_dt / substep_dt)
        output_dir = self.cfg.output
        camera_mode = self.cfg.camera.camera_mode
        cameras_json = self.cfg.camera.cameras_json
        if cameras_json and not os.path.isabs(cameras_json):
            cameras_json = os.path.join(self.config_dir, cameras_json)

        print("Running simulation and rendering...")
        print(
            f"  substep_dt={substep_dt:.2e}  frame_dt={frame_dt:.2e}  "
            f"steps/frame={step_per_frame}  frames={frame_num}"
        )

        stale = sorted(_glob.glob(os.path.join(output_dir, "[0-9]*.png")))
        if stale:
            expected = {
                os.path.join(output_dir, f"{i:04d}.png")
                for i in range(frame_num)
            }
            to_remove = [p for p in stale if p not in expected]
            if to_remove:
                print(f"  Removing {len(to_remove)} stale frame PNG(s)")
                for p in to_remove:
                    os.remove(p)

        bg_color = (
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            if self.args.white_bg
            else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        )
        opacity_render = self.sim_opacity
        shs_render = self.sim_shs
        has_static = len(self.static_chunks) > 0
        height = width = None

        for frame in tqdm(range(frame_num), desc="Simulating"):
            if camera_mode == "json":
                camera = self.renderer.build_camera_from_json(
                    cameras_json,
                    self.camera_params,
                    center_view_world_space=self.viewpoint_center_worldspace,
                    observant_coordinates=self.observant_coordinates,
                    current_frame=frame,
                )
            elif camera_mode == "orbit":
                camera = self.renderer.build_camera_orbit(
                    camera_params=self.camera_params,
                    center_view_world_space=self.viewpoint_center_worldspace,
                    observant_coordinates=self.observant_coordinates,
                    current_frame=frame,
                )
            else:
                camera = self.renderer.build_camera_fixed(
                    camera_params=self.camera_params,
                )

            for _ in range(step_per_frame):
                self.backend.step(substep_dt, frame)

            state = self.backend.get_state()
            pos = state.positions[: self.gs_num].to(self.device)
            cov3D = state.covariances[: self.gs_num].to(self.device)
            rot = state.rotations[: self.gs_num].to(self.device)

            if hasattr(self.backend, "get_diagnostics"):
                diag = self.backend.get_diagnostics()
                for bd in diag["bodies"]:
                    p, v = bd["pos"], bd["vel"]
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

            nan_mask = ~torch.isfinite(pos).all(dim=1)
            extreme_mask = (pos.abs() > 100.0).any(dim=1)
            bad_mask = nan_mask | extreme_mask
            if bad_mask.any():
                n_nan = int(nan_mask.sum().item())
                n_ext = int((extreme_mask & ~nan_mask).sum().item())
                print(
                    f"[WARNING] Frame {frame}: "
                    f"{n_nan} NaN/Inf + {n_ext} extreme particles"
                )
                pos[bad_mask] = 0.0
                cov3D[bad_mask] = 0.0

            pos = apply_inverse_rotations(pos, self.rotation_matrices)
            cov3D = apply_inverse_cov_rotations(cov3D, self.rotation_matrices)

            render_quats = render_scales = None
            if self.gs_type == "2dgs" and state.quats is not None:
                render_quats = apply_axis_perm_to_quats(
                    inverse_preprocess_quats(
                        state.quats[: self.gs_num].to(self.device),
                        self.axis_perm,
                        self.rotation_matrices,
                    ),
                    self.axis_perm,
                )
                render_scales = state.scales[: self.gs_num].to(self.device)

            cur_opacity = opacity_render
            cur_shs = shs_render
            if has_static:
                pos = torch.cat([pos, self.static_pos], dim=0)
                cov3D = torch.cat([cov3D, self.static_cov], dim=0)
                cur_opacity = torch.cat(
                    [opacity_render, self.static_opacity], dim=0
                )
                cur_shs = torch.cat([shs_render, self.static_shs], dim=0)
                if render_quats is not None and self.static_quats is not None:
                    render_quats = torch.cat(
                        [
                            render_quats,
                            apply_axis_perm_to_quats(
                                self.static_quats, self.axis_perm,
                            ),
                        ],
                        dim=0,
                    )
                    render_scales = torch.cat(
                        [render_scales, self.static_scales], dim=0
                    )

            colors_precomp = self.renderer.convert_sh(
                cur_shs, camera, pos, rot,
            )
            if self.gs_type == "2dgs" and render_quats is not None:
                rendering, _meta = self.renderer.render(
                    camera=camera,
                    means=pos,
                    colors=colors_precomp,
                    opacities=cur_opacity,
                    bg_color=bg_color,
                    quats=render_quats,
                    scales=render_scales,
                )
            else:
                rendering, _meta = self.renderer.render(
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

        if self.args.compile_video:
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


# ── CLI entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3DGS Physics Simulation Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument(
        "--no_render", action="store_true", help="Run physics only, skip rendering."
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree of PLY models"
    )
    parser.add_argument(
        "--raster_backend",
        type=str,
        default="gsplat",
        choices=["gsplat", "diffrast"],
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print intermediate tensor stats"
    )
    parser.add_argument("--debug_log", type=str, default=None)
    raw = parser.parse_args()

    assert os.path.exists(raw.config), f"Config not found: {raw.config}"

    args = PipelineArgs(
        config=raw.config,
        no_render=raw.no_render,
        white_bg=raw.white_bg,
        compile_video=raw.compile_video,
        debug=raw.debug,
        debug_log=raw.debug_log,
        sh_degree=raw.sh_degree,
        raster_backend=raw.raster_backend,
    )

    print("Loading config...")
    cfg = ConfigLoader().load(raw.config)
    config_dir = os.path.dirname(os.path.abspath(raw.config))

    pipeline = SimulationPipeline(cfg, args)
    pipeline.run(config_dir=config_dir)


if __name__ == "__main__":
    main()
