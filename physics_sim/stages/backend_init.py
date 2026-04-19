"""Stage 2: Backend initialization, material setup, and boundary conditions.

All positions, directions, and boundary conditions are already in the
internal Y-up coordinate system after :func:`setup_scene`.  This stage
transforms user-specified "world" BCs (which are in source coordinates)
into internal coordinates using :mod:`physics_sim.coord`.

Gravity
-------
Direction is always ``[0, -g_magnitude, 0]`` in internal Y-up space.
The user only specifies ``g_magnitude`` (default 9.8) in the material
dict. ``g`` is still accepted for compatibility:
- scalar ``g`` is normalized to ``[0, -|g|, 0]`` with context log;
- vector ``g`` must already satisfy the internal Y-up contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from physics_sim.backend.registry import create_backend, resolve_material
from physics_sim.config.models import SimConfig
from physics_sim.coord import (
    E_GRAVITY_SHAPE,
    SourceAxes,
    align_directions,
    align_positions,
    gravity_contract_error,
    gravity_vector,
    normalize_internal_gravity,
)
from physics_sim.geometry.plane_fit import fit_plane_svd
from physics_sim.logging_utils import get_logger

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.stages.scene_setup import SceneData

LOGGER = get_logger(__name__)


def _resolve_gravity(per_object_info: list[dict]) -> list[float]:
    """Determine gravity vector in internal Y-up coordinates.

    Contract:
    - ``g_magnitude``: scalar magnitude, direction fixed to -Y.
    - ``g`` scalar: accepted and normalized to ``[0, -|g|, 0]`` with context log.
    - ``g`` vector: must already satisfy internal Y-up ``[0, -|g|, 0]``.
    """
    default_magnitude = 9.8
    for idx, info in enumerate(per_object_info):
        mat = info.get("material", {})
        material_name = info.get("name", f"per_object[{idx}]")
        cfg_path = f"per_object[{idx}].material.g"

        if "g_magnitude" in mat:
            try:
                magnitude = abs(float(mat["g_magnitude"]))
            except (TypeError, ValueError) as exc:
                raise gravity_contract_error(
                    E_GRAVITY_SHAPE,
                    backend="backend_init",
                    config_path=f"per_object[{idx}].material.g_magnitude",
                    material_name=material_name,
                    raw_g=mat.get("g_magnitude"),
                    detail="g_magnitude must be a numeric scalar",
                    suggestion="set g_magnitude to a finite number, e.g. 9.8",
                ) from exc
            if not np.isfinite(magnitude):
                raise gravity_contract_error(
                    E_GRAVITY_SHAPE,
                    backend="backend_init",
                    config_path=f"per_object[{idx}].material.g_magnitude",
                    material_name=material_name,
                    raw_g=mat.get("g_magnitude"),
                    detail="g_magnitude must be finite",
                    suggestion="set g_magnitude to a finite number, e.g. 9.8",
                )
            return gravity_vector(magnitude, device="cpu").tolist()

        if "g" in mat:
            raw_g = mat.get("g")
            resolved_g = normalize_internal_gravity(
                raw_g,
                backend="backend_init",
                config_path=cfg_path,
                allow_scalar=True,
                material_name=material_name,
            )
            if isinstance(raw_g, (int, float)):
                LOGGER.info(
                    "[Gravity][backend=backend_init]"
                    "[material=%s]"
                    "[config_path=%s]"
                    " scalar_g=%r resolved_g=%s",
                    material_name,
                    cfg_path,
                    raw_g,
                    list(resolved_g),
                )
            return list(resolved_g)

    return gravity_vector(default_magnitude, device="cpu").tolist()


def _resolve_collider_bc(
    collider_objects,
    source_axes: SourceAxes,
) -> list[dict]:
    """Convert collider_only objects into surface_collider BCs.

    Fitted planes are computed from positions already in internal Y-up
    space, so their point/normal are also in Y-up.
    """
    bc_list: list[dict] = []
    for obj in collider_objects:
        col = obj.collider or {}
        col_type = col.get("type", "plane")
        if col_type != "plane":
            raise ValueError(
                f"collider_only '{obj.name}' only supports type='plane' "
                f"(got {col_type!r})"
            )

        surface = col.get("surface", "sticky")
        friction = float(col.get("friction", 0.0))
        start_time = col.get("start_time", 0)
        end_time = col.get("end_time", 1e3)

        if col.get("point") is not None and col.get("normal") is not None:
            pt_src = torch.tensor(col["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
            nr_src = torch.tensor(col["normal"], device="cuda", dtype=torch.float32).reshape(1, 3)
            point = align_positions(pt_src, source_axes)[0]
            normal = align_directions(nr_src, source_axes)[0]
            normal = normal / (torch.norm(normal) + 1e-12)
            point = [float(x) for x in point.cpu().tolist()]
            normal = [float(x) for x in normal.cpu().tolist()]
        else:
            pts = obj.positions.detach().cpu().numpy()
            if pts.shape[0] < 3:
                raise ValueError(
                    f"collider_only '{obj.name}' has too few points for "
                    "plane fitting and no explicit point+normal."
                )
            fit = col.get("fit") or {}
            sample_max = fit.get("sample_max", 200000)
            # prefer_up: transform from source to internal
            prefer_up_src = col.get("prefer_up", source_axes.up_vector.tolist())
            prefer_up_t = torch.tensor(prefer_up_src, dtype=torch.float32).reshape(1, 3)
            prefer_up_aligned = align_directions(prefer_up_t, source_axes)[0].numpy()
            res = fit_plane_svd(
                pts,
                sample_max=sample_max,
                seed=int(fit.get("seed", 0)),
                prefer_up=np.asarray(prefer_up_aligned, dtype=np.float32),
            )
            point = res.point.tolist()
            normal = res.normal.tolist()
            LOGGER.info(
                "[collider] name=%s plane_rms=%.6f point=%s normal=%s",
                obj.name,
                res.rms,
                point,
                normal,
            )

        bc_list.append(dict(
            type="surface_collider",
            space="internal",
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
    source_axes = scene_data.source_axes
    LOGGER.info("Initialising backend: %s", bt)

    backend = create_backend(backend_cfg, device=device)

    init_kwargs = backend_cfg.model_dump(exclude={"type"}, exclude_none=True)

    if scene_data.gs_type == "2dgs" and scene_data.gs_num > 0:
        init_kwargs["init_quats"] = scene_data.sim_quats
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

    if scene_data.per_object_info:
        resolved_info = []
        for info in scene_data.per_object_info:
            resolved = dict(info)
            resolved["material"] = resolve_material(bt, info["material"])
            resolved_info.append(resolved)
        material_params["per_object"] = resolved_info

    # Gravity: always -Y in internal Y-up space; resolved once at stage boundary.
    material_params["g"] = _resolve_gravity(scene_data.per_object_info)

    backend.set_material(material_params)

    # ── Boundary conditions ───────────────────────────────────────────
    bc_all: list[dict] = []
    for bc in cfg.boundary_conditions:
        bc_all.append(bc.model_dump() if hasattr(bc, "model_dump") else dict(bc))

    collider_bcs = _resolve_collider_bc(scene_data.collider_objects, source_axes)
    bc_all.extend(collider_bcs)

    # Transform user-specified "world" BCs from source coords to internal
    bc_converted: list[dict] = []
    for bc in bc_all:
        space = bc.get("space", "")
        if space == "world" and bc.get("type") == "surface_collider":
            p_src = torch.tensor(bc["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
            n_src = torch.tensor(bc["normal"], device="cuda", dtype=torch.float32).reshape(1, 3)
            p_int = align_positions(p_src, source_axes)[0]
            n_int = align_directions(n_src, source_axes)[0]
            n_int = n_int / (torch.norm(n_int) + 1e-12)
            bc_new = dict(bc)
            bc_new["point"] = [float(x) for x in p_int.cpu().tolist()]
            bc_new["normal"] = [float(x) for x in n_int.cpu().tolist()]
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
