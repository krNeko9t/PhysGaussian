"""Scene assembler: reads config + PLY files and produces list[SceneObject].

Consumes the Pydantic-based :class:`~physics_sim.config.models.SimConfig`
where each object is a typed ``ObjectConfig``.

Coordinate alignment
--------------------
Source data is aligned to the internal **Y-up** convention via
:mod:`physics_sim.coord`.  The ``source_up`` / ``source_front`` fields in
``PreprocessConfig`` declare the PLY coordinate system.  All position,
covariance, and quaternion data are transformed **once** in this module;
downstream code always sees Y-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from physics_sim.config.models import (
    IdMapSource,
    ObjectConfig,
    PlySource,
    SimConfig,
)
from physics_sim.coord import (
    SourceAxes,
    align_covariances,
    align_positions,
    align_quats,
)
from physics_sim.scene.objects import SceneObject


# ── Helpers ───────────────────────────────────────────────────────────

def _resolve_path(path: str, config_dir: str | None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if config_dir is not None:
        candidate = Path(config_dir) / p
        if candidate.exists():
            return str(candidate)
    return str(p)


# ── Public API ────────────────────────────────────────────────────────

def assemble_scene(
    cfg: SimConfig,
    renderer: Any,
    config_dir: str | None = None,
) -> list[SceneObject]:
    """Build a list of :class:`SceneObject` from a :class:`SimConfig`.

    All returned objects have positions/covariances/quaternions in the
    internal Y-up coordinate system.
    """
    pp = cfg.preprocess
    source_axes = SourceAxes.from_config(pp.source_up, pp.source_front)
    global_opacity_threshold = pp.opacity_threshold

    if not cfg.objects:
        raise ValueError("Config must declare at least one object.")

    ply_cache: dict[str, dict] = {}
    id_map_cache: dict[str, np.ndarray] = {}

    def _load_ply(path: str) -> dict:
        resolved = _resolve_path(path, config_dir)
        if resolved not in ply_cache:
            print(f"  [assembler] Loading PLY: {resolved}")
            ply_cache[resolved] = renderer.load_ply(resolved)
        return ply_cache[resolved]

    def _load_id_map(path: str) -> np.ndarray:
        resolved = _resolve_path(path, config_dir)
        if resolved not in id_map_cache:
            print(f"  [assembler] Loading ID map: {resolved}")
            id_map_cache[resolved] = np.load(resolved)
        return id_map_cache[resolved]

    result: list[SceneObject] = []

    for obj_cfg in cfg.objects:
        name = obj_cfg.name
        role = obj_cfg.role
        source = obj_cfg.source

        print(f"  [assembler] Processing '{name}' (role={role}, source={source.type})")

        # ── Load GS data ──────────────────────────────────────────────
        if isinstance(source, PlySource):
            ply_data = _load_ply(source.ply_path)
            pos = ply_data["pos"]
            cov = ply_data["cov3D_precomp"]
            opacity = ply_data["opacity"]
            shs = ply_data["shs"]
            quats = ply_data["quats"]
            scales = ply_data["scales"]
            gs_type = ply_data["gs_type"]

        elif isinstance(source, IdMapSource):
            ply_data = _load_ply(source.ply_path)
            id_map = _load_id_map(source.id_map)
            id_mask = torch.from_numpy(id_map == source.object_id).to(
                device=ply_data["pos"].device,
            )
            pos = ply_data["pos"][id_mask]
            cov = ply_data["cov3D_precomp"][id_mask]
            opacity = ply_data["opacity"][id_mask]
            shs = ply_data["shs"][id_mask]
            quats = ply_data["quats"][id_mask]
            scales = ply_data["scales"][id_mask]
            gs_type = ply_data["gs_type"]

        else:
            raise ValueError(f"Object '{name}': unsupported source type {type(source)}")

        # ── Opacity filtering ─────────────────────────────────────────
        opacity_threshold = (
            obj_cfg.opacity_threshold
            if obj_cfg.opacity_threshold is not None
            else global_opacity_threshold
        )
        op_mask = opacity[:, 0] > opacity_threshold
        pos = pos[op_mask]
        cov = cov[op_mask]
        opacity = opacity[op_mask]
        shs = shs[op_mask]
        quats = quats[op_mask]
        scales = scales[op_mask]

        # ── Position offset (in source coordinates) ───────────────────
        offset = obj_cfg.transform.position
        if any(v != 0.0 for v in offset):
            pos = pos + torch.tensor(offset, device=pos.device, dtype=pos.dtype)

        # ── Coordinate alignment (source -> internal Y-up) ───────────
        pos = align_positions(pos, source_axes)
        cov = align_covariances(cov, source_axes)
        quats = align_quats(quats, source_axes)

        # ── Collider / filling dicts for downstream ───────────────────
        collider_dict = obj_cfg.collider.model_dump() if obj_cfg.collider else None
        filling_dict = (
            obj_cfg.particle_filling.model_dump()
            if obj_cfg.particle_filling
            else None
        )

        print(f"    -> {name}: {pos.shape[0]} GS particles (gs_type={gs_type})")

        result.append(SceneObject(
            name=name,
            role=role,
            positions=pos,
            covariances=cov,
            opacities=opacity,
            shs=shs,
            quats=quats,
            scales=scales,
            material=dict(obj_cfg.material),
            initial_velocity=obj_cfg.initial_velocity,
            particle_filling=filling_dict,
            collider=collider_dict,
            gs_type=gs_type,
        ))

    return result
