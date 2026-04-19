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

from physics_sim.backend.registry import create_backend, resolve_material
from physics_sim.config.models import SimConfig
from physics_sim.coord import (
    E_GRAVITY_SHAPE,
    gravity_contract_error,
    gravity_vector,
    normalize_internal_gravity,
)
from physics_sim.logging_utils import get_logger
from physics_sim.stages.boundary_normalization import normalize_boundary_conditions

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


def _build_material_params(cfg: SimConfig, scene_data: SceneData) -> dict:
    backend_cfg = cfg.backend
    backend_type = backend_cfg.type
    params: dict = {}
    params.setdefault("n_grid", getattr(backend_cfg, "n_grid", 200))
    params.setdefault("grid_lim", getattr(backend_cfg, "grid_lim", 2.0))

    if hasattr(backend_cfg, "solver_iterations") and backend_cfg.solver_iterations is not None:
        params["newton_solver_opts"] = {"iterations": backend_cfg.solver_iterations}
        if hasattr(backend_cfg, "contact_relaxation"):
            params["newton_solver_opts"]["contact_relaxation"] = backend_cfg.contact_relaxation

    if scene_data.per_object_info:
        resolved_info = []
        for info in scene_data.per_object_info:
            resolved = dict(info)
            resolved["material"] = resolve_material(backend_type, info["material"])
            resolved_info.append(resolved)
        params["per_object"] = resolved_info

    # Gravity is normalized exactly once at stage boundary.
    params["g"] = _resolve_gravity(scene_data.per_object_info)
    return params


def _collect_raw_boundary_conditions(cfg: SimConfig) -> list[dict]:
    raw: list[dict] = []
    for bc in cfg.boundary_conditions:
        raw.append(bc.model_dump() if hasattr(bc, "model_dump") else dict(bc))
    return raw


def _build_time_params(cfg: SimConfig) -> dict[str, float | int]:
    tc = cfg.time
    return {
        "substep_dt": tc.substep_dt,
        "frame_dt": tc.frame_dt,
        "frame_num": tc.frame_num,
    }


def init_backend(
    cfg: SimConfig,
    scene_data: SceneData,
    device: str = "cuda:0",
) -> PhysicsBackend:
    """Create, configure, and finalize the physics backend."""
    backend_cfg = cfg.backend
    bt = backend_cfg.type
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

    material_params = _build_material_params(cfg, scene_data)
    backend.set_material(material_params)

    raw_bc = _collect_raw_boundary_conditions(cfg)
    normalized_bc = normalize_boundary_conditions(
        raw_boundary_conditions=raw_bc,
        collider_objects=scene_data.collider_objects,
        source_axes=scene_data.source_axes,
    )
    backend.set_boundary_conditions(normalized_bc, _build_time_params(cfg))
    backend.finalize()

    return backend
