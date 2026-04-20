"""Stage 2: Backend initialization, material setup, and boundary conditions.

All positions, directions, and boundary conditions are already in the
internal Y-up coordinate system after :func:`setup_scene`.  This stage
transforms user-specified "world" BCs (which are in source coordinates)
into internal coordinates using :mod:`physics_sim.coord`.

Gravity
-------
Direction is always ``[0, -g_magnitude, 0]`` in internal Y-up space.
Each :class:`MaterialSpec` carries its own ``g_magnitude`` (default 9.8);
this stage reads the first dynamic object's value and forms the vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from physics_sim.backend.registry import create_backend
from physics_sim.config.models import SimConfig
from physics_sim.coord import (
    E_GRAVITY_SHAPE,
    gravity_contract_error,
    gravity_vector,
)
from physics_sim.logging_utils import get_logger
from physics_sim.stages.boundary_normalization import normalize_boundary_conditions

if TYPE_CHECKING:
    from physics_sim.backend.base import PhysicsBackend
    from physics_sim.stages.scene_setup import ObjectRuntimeInfo, SceneData

LOGGER = get_logger(__name__)


def _resolve_gravity(objects_runtime: list[ObjectRuntimeInfo]) -> list[float]:
    """Read ``g_magnitude`` from the first dynamic object and form ``[0, -g, 0]``."""
    default_magnitude = 9.8
    if not objects_runtime:
        return gravity_vector(default_magnitude, device="cpu").tolist()

    info = objects_runtime[0]
    magnitude = abs(float(info.material.g_magnitude))
    if not np.isfinite(magnitude):
        raise gravity_contract_error(
            E_GRAVITY_SHAPE,
            backend="backend_init",
            config_path=f"objects_runtime[0].material.g_magnitude",
            material_name=info.name,
            raw_g=info.material.g_magnitude,
            detail="g_magnitude must be finite",
            suggestion="set g_magnitude to a finite number, e.g. 9.8",
        )
    return gravity_vector(magnitude, device="cpu").tolist()


def _build_material_params(cfg: SimConfig, scene_data: SceneData) -> dict:
    backend_cfg = cfg.backend
    params: dict = {}
    params.setdefault("n_grid", getattr(backend_cfg, "n_grid", 200))
    params.setdefault("grid_lim", getattr(backend_cfg, "grid_lim", 2.0))

    if hasattr(backend_cfg, "solver_iterations") and backend_cfg.solver_iterations is not None:
        params["newton_solver_opts"] = {"iterations": backend_cfg.solver_iterations}
        if hasattr(backend_cfg, "contact_relaxation"):
            params["newton_solver_opts"]["contact_relaxation"] = backend_cfg.contact_relaxation

    if scene_data.objects_runtime:
        params["per_object"] = list(scene_data.objects_runtime)

    # Gravity is normalized exactly once at stage boundary.
    params["g"] = _resolve_gravity(scene_data.objects_runtime)
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
    dyn = scene_data.dynamic_init
    if scene_data.gs_type == "2dgs" and scene_data.gs_num > 0:
        init_kwargs["init_quats"] = dyn.quats
        init_kwargs["init_scales"] = dyn.scales
    if scene_data.gs_num > 0:
        backend.initialize(
            dyn.pos,
            dyn.vol,
            dyn.cov,
            **init_kwargs,
        )

    material_params = _build_material_params(cfg, scene_data)
    backend.set_material(material_params)

    raw_bc = _collect_raw_boundary_conditions(cfg)
    normalized_bc = normalize_boundary_conditions(
        raw_boundary_conditions=raw_bc,
        collider_objects=scene_data.collider_objects,
        source_axes=scene_data.coord.source_axes,
    )
    backend.set_boundary_conditions(normalized_bc, _build_time_params(cfg))
    backend.finalize()

    return backend

