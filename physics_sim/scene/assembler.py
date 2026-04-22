"""Scene assembler: reads config + PLY files and produces list[SceneObject].

Consumes ``cfg.scene`` (a ``SceneConfig`` with ``parts`` and
``constraints``) from :class:`~physics_sim.config.models.SimConfig`.

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

import numpy as np
import torch

from physics_sim.config.models import (
    IdMapSource,
    PointSelectorSource,
    PlySource,
    SimConfig,
)
from physics_sim.config.scene import CollideOnly
from physics_sim.render.interfaces import SceneAssetLoader
from physics_sim.render.types import GaussianAsset
from physics_sim.coord import (
    SourceAxes,
    align_covariances,
    align_positions,
    align_quats,
)
from physics_sim.scene.objects import SceneObject
from physics_sim.scene.point_selector import PointSelectorResolver


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


def _ctx(object_name: str, source_type: str) -> str:
    return f"[assembler][object={object_name}][source={source_type}]"


# ── Public API ────────────────────────────────────────────────────────

def assemble_scene(
    cfg: SimConfig,
    loader: SceneAssetLoader,
    config_dir: str | None = None,
) -> list[SceneObject]:
    """Build a list of :class:`SceneObject` from a :class:`SimConfig`.

    All returned objects have positions/covariances/quaternions in the
    internal Y-up coordinate system.
    """
    pp = cfg.preprocess
    source_axes = SourceAxes.from_config(pp.source_up, pp.source_front)
    global_opacity_threshold = pp.opacity_threshold

    scene = cfg.scene
    if not scene.parts:
        raise ValueError("Config must declare at least one part.")

    # Derive per-part role from scene + material.  A part referenced
    # by a CollideOnly constraint becomes "collider_only"; a part with
    # no material is "render_only"; everything else is "dynamic".
    collide_only_names = {
        c.part for c in scene.constraints if isinstance(c, CollideOnly)
    }

    ply_cache: dict[str, GaussianAsset] = {}
    id_map_cache: dict[str, np.ndarray] = {}
    selector_resolver = PointSelectorResolver(config_dir=config_dir)

    def _load_ply(path: str) -> GaussianAsset:
        resolved = _resolve_path(path, config_dir)
        if resolved not in ply_cache:
            print(f"  [assembler] Loading PLY: {resolved}")
            ply_cache[resolved] = loader.load_ply(resolved)
        return ply_cache[resolved]

    def _resolve_for_object(path: str, object_name: str, source_type: str) -> str:
        resolved = _resolve_path(path, config_dir)
        if not Path(resolved).exists():
            raise FileNotFoundError(
                f"{_ctx(object_name, source_type)} missing file: {resolved}",
            )
        return resolved

    def _load_id_map(path: str, *, object_name: str, source_type: str) -> np.ndarray:
        resolved = _resolve_for_object(path, object_name=object_name, source_type=source_type)
        if resolved not in id_map_cache:
            print(f"  [assembler] Loading ID map: {resolved}")
            id_map_cache[resolved] = np.load(resolved)
        return id_map_cache[resolved]

    result: list[SceneObject] = []

    for obj_cfg in scene.parts:
        name = obj_cfg.name
        if name in collide_only_names:
            role = "collider_only"
        elif obj_cfg.material is None:
            role = "render_only"
        else:
            role = "dynamic"
        source = obj_cfg.source

        print(f"  [assembler] Processing '{name}' (role={role}, source={source.type})")

        # ── Load GS data ──────────────────────────────────────────────
        if isinstance(source, PlySource):
            ply_data = _load_ply(source.ply_path)
            pos = ply_data.pos
            cov = ply_data.cov3D_precomp
            opacity = ply_data.opacity
            shs = ply_data.shs
            quats = ply_data.quats
            scales = ply_data.scales
            gs_type = ply_data.gs_type

        elif isinstance(source, IdMapSource):
            ply_data = _load_ply(source.ply_path)
            id_map = _load_id_map(source.id_map, object_name=name, source_type=source.type)
            if id_map.ndim != 1:
                raise ValueError(
                    f"{_ctx(name, source.type)} id_map must be 1D, got shape={id_map.shape} "
                    f"for {source.id_map}",
                )
            if id_map.shape[0] != ply_data.pos.shape[0]:
                raise ValueError(
                    f"{_ctx(name, source.type)} id_map length mismatch: "
                    f"id_map={id_map.shape[0]}, ply_gs={ply_data.pos.shape[0]}, path={source.id_map}",
                )
            id_mask = torch.from_numpy(id_map == source.object_id).to(
                device=ply_data.pos.device,
            )
            pos = ply_data.pos[id_mask]
            cov = ply_data.cov3D_precomp[id_mask]
            opacity = ply_data.opacity[id_mask]
            shs = ply_data.shs[id_mask]
            quats = ply_data.quats[id_mask]
            scales = ply_data.scales[id_mask]
            gs_type = ply_data.gs_type

        elif isinstance(source, PointSelectorSource):
            base_resolved = _resolve_for_object(
                source.base_ply_path,
                object_name=name,
                source_type=source.type,
            )
            ply_data = _load_ply(base_resolved)
            base_positions = ply_data.pos.detach().cpu().numpy().astype(np.float64, copy=False)
            final_mask_np = selector_resolver.build_mask(
                object_name=name,
                source=source,
                base_path=base_resolved,
                base_positions=base_positions,
            )

            select_mask = torch.from_numpy(final_mask_np).to(
                device=ply_data.pos.device,
            )
            pos = ply_data.pos[select_mask]
            cov = ply_data.cov3D_precomp[select_mask]
            opacity = ply_data.opacity[select_mask]
            shs = ply_data.shs[select_mask]
            quats = ply_data.quats[select_mask]
            scales = ply_data.scales[select_mask]
            gs_type = ply_data.gs_type

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

        # ── Collider / filling objects propagate typed ──────────────

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
            material=obj_cfg.material,
            initial_velocity=obj_cfg.initial_velocity,
            particle_filling=obj_cfg.particle_filling,
            fill_group=obj_cfg.fill_group,
            collider=obj_cfg.collider,
            gs_type=gs_type,
        ))

    return result
