"""Stage 2: Backend initialization, material setup, and boundary conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from physics_sim.backend.registry import create_backend, resolve_material
from physics_sim.config.models import SimConfig
from physics_sim.geometry.plane_fit import fit_plane_svd
from physics_sim.preprocessing.transform import apply_rotations
from physics_sim.preprocessing.quaternions import preprocess_quats

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.stages.scene_setup import SceneData


def _resolve_collider_bc(
    collider_objects,
    axis_perm: str,
    opacity_threshold: float,
) -> list[dict]:
    """Convert collider_only objects into surface_collider BCs."""
    bc_list: list[dict] = []
    for obj in collider_objects:
        col = obj.collider or {}
        col_type = col.get("type", "plane")
        if col_type != "plane":
            raise ValueError(
                f"collider_only '{obj.name}' only supports type='plane' "
                f"(got {col_type!r})"
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
                    f"collider_only '{obj.name}' has too few points for "
                    "plane fitting and no explicit point+normal."
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


def init_backend(
    cfg: SimConfig,
    scene_data: SceneData,
    device: str = "cuda:0",
) -> PhysicsBackend:
    """Create, configure, and finalize the physics backend."""
    backend_cfg = cfg.backend
    bt = backend_cfg.type
    print(f"Initialising backend: {bt}")

    backend = create_backend(backend_cfg, device=device)

    # Collect backend-specific kwargs from the typed config
    init_kwargs = backend_cfg.model_dump(exclude={"type"}, exclude_none=True)

    if scene_data.gs_type == "2dgs" and scene_data.gs_num > 0:
        init_kwargs["init_quats"] = preprocess_quats(
            scene_data.sim_quats,
            scene_data.axis_perm,
            scene_data.rotation_matrices,
        )
        init_kwargs["init_scales"] = scene_data.sim_scales

    if scene_data.gs_num > 0:
        backend.initialize(
            scene_data.sim_init_pos,
            scene_data.sim_init_vol,
            scene_data.sim_init_cov,
            **init_kwargs,
        )

    # ── Material ──────────────────────────────────────────────────────
    material_params: dict = {}
    material_params.setdefault("n_grid", getattr(backend_cfg, "n_grid", 200))
    material_params.setdefault("grid_lim", getattr(backend_cfg, "grid_lim", 2.0))

    if hasattr(backend_cfg, "solver_iterations") and backend_cfg.solver_iterations is not None:
        material_params["newton_solver_opts"] = {
            "iterations": backend_cfg.solver_iterations,
        }
        if hasattr(backend_cfg, "contact_relaxation"):
            material_params["newton_solver_opts"]["contact_relaxation"] = (
                backend_cfg.contact_relaxation
            )

    # Resolve per-object materials with backend defaults
    if scene_data.per_object_info:
        resolved_info = []
        for info in scene_data.per_object_info:
            resolved = dict(info)
            resolved["material"] = resolve_material(bt, info["material"])
            resolved_info.append(resolved)
        material_params["per_object"] = resolved_info

    # Extract gravity from any object's material (first one that has 'g')
    for info in scene_data.per_object_info:
        if "g" in info["material"]:
            material_params.setdefault("g", info["material"]["g"])
            break

    backend.set_material(material_params)

    # ── Boundary conditions ───────────────────────────────────────────
    bc_all: list[dict] = []
    for bc in cfg.boundary_conditions:
        bc_all.append(bc.model_dump() if hasattr(bc, "model_dump") else dict(bc))

    collider_bcs = _resolve_collider_bc(
        scene_data.collider_objects,
        scene_data.axis_perm,
        cfg.preprocess.opacity_threshold,
    )
    bc_all.extend(collider_bcs)

    # Convert world-space colliders to rotated space
    bc_converted: list[dict] = []
    for bc in bc_all:
        if bc.get("space") == "world" and bc.get("type") == "surface_collider":
            p_w = torch.tensor(bc["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
            n_w = torch.tensor(bc["normal"], device="cuda", dtype=torch.float32).reshape(1, 3)
            p_r = apply_rotations(p_w, scene_data.rotation_matrices)[0]
            n_r = apply_rotations(n_w, scene_data.rotation_matrices)[0]
            n_r = n_r / (torch.norm(n_r) + 1e-12)
            bc_new = dict(bc)
            bc_new["point"] = [float(x) for x in p_r.detach().cpu().tolist()]
            bc_new["normal"] = [float(x) for x in n_r.detach().cpu().tolist()]
            bc_new.pop("space", None)
            bc_converted.append(bc_new)
        else:
            bc_converted.append(bc)

    tc = cfg.time
    time_params = {
        "substep_dt": tc.substep_dt,
        "frame_dt": tc.frame_dt,
        "frame_num": tc.frame_num,
    }
    backend.set_boundary_conditions(bc_converted, time_params)
    backend.finalize()

    return backend
