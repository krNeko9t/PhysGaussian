"""Scene assembler: reads config + PLY files and produces list[SceneObject].

This module replaces the ~350 lines of inline object-assembly logic that
previously lived in ``pipeline.py`` (lines 490-848 of the old code).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physics_sim.config.schema import SimConfig
from physics_sim.scene.objects import SceneObject


def _apply_axis_permutation(
    pos: torch.Tensor, cov: torch.Tensor, perm: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis permutation on positions and covariances."""
    if perm == "xyz":
        return pos, cov

    axis_map = {"x": 0, "y": 1, "z": 2}
    idx: list[int] = []
    signs: list[float] = []
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
        [cov_perm[:, 0, 0], cov_perm[:, 0, 1], cov_perm[:, 0, 2],
         cov_perm[:, 1, 1], cov_perm[:, 1, 2], cov_perm[:, 2, 2]],
        dim=1,
    )
    return pos_new, cov_new


def _apply_sim_area_mask(
    positions: torch.Tensor, boundary: list[float],
) -> torch.Tensor:
    """Return a boolean mask selecting particles inside an AABB."""
    assert len(boundary) == 6
    mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
    for i in range(3):
        mask &= positions[:, i] > boundary[2 * i]
        mask &= positions[:, i] < boundary[2 * i + 1]
    return mask


def _merge_material(top_level: dict, per_object: dict | None) -> dict:
    """Deep-merge top-level material defaults with per-object overrides."""
    merged = copy.deepcopy(top_level)
    if per_object:
        merged.update(per_object)
    return merged


def _resolve_path(path: str, config_dir: str | None) -> str:
    """Resolve a potentially relative path against the config directory."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if config_dir is not None:
        candidate = Path(config_dir) / p
        if candidate.exists():
            return str(candidate)
    return str(p)


def assemble_scene(
    cfg: SimConfig,
    renderer: Any,
    config_dir: str | None = None,
) -> list[SceneObject]:
    """Build a list of SceneObject from a fully-resolved :class:`SimConfig`.

    Args:
        cfg: Fully resolved config (all ``_ref`` references expanded).
        renderer: A ``GaussianRenderer`` instance (used for ``load_ply``).
        config_dir: Directory of the config file, used for resolving
            relative PLY paths.

    Returns:
        List of ``SceneObject`` instances, one per declared object.
    """
    from physics_sim.preprocessing.transform import (
        generate_rotation_matrices,
        apply_rotations,
        apply_cov_rotations,
    )

    pp = cfg.preprocess
    axis_perm = pp.axis_permutation
    opacity_threshold = pp.opacity_threshold

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(pp.rotation_degree), pp.rotation_axis,
    )

    top_material = dict(cfg.material)

    objects_cfg = cfg.objects
    if not objects_cfg:
        raise ValueError("Config must declare at least one object in 'objects'.")

    ply_cache: dict[str, dict] = {}
    id_map_cache: dict[str, np.ndarray] = {}

    def _load_ply_cached(path: str) -> dict:
        resolved = _resolve_path(path, config_dir)
        if resolved not in ply_cache:
            print(f"  [assembler] Loading PLY: {resolved}")
            ply_cache[resolved] = renderer.load_ply(resolved)
        return ply_cache[resolved]

    def _load_id_map_cached(path: str) -> np.ndarray:
        resolved = _resolve_path(path, config_dir)
        if resolved not in id_map_cache:
            print(f"  [assembler] Loading ID map: {resolved}")
            id_map_cache[resolved] = np.load(resolved)
        return id_map_cache[resolved]

    result: list[SceneObject] = []

    for i, obj_cfg in enumerate(objects_cfg):
        name = obj_cfg.get("name", f"object_{i}")
        mode = obj_cfg.get("mode", "simulate")
        source = obj_cfg.get("source", {})
        source_type = source.get("type", "ply")

        print(f"  [assembler] Processing '{name}' (mode={mode}, source={source_type})")

        if source_type == "ply":
            ply_path = source.get("ply_path")
            if ply_path is None:
                raise ValueError(f"Object '{name}': source.type='ply' requires 'ply_path'.")
            ply_data = _load_ply_cached(ply_path)
            pos = ply_data["pos"]
            cov = ply_data["cov3D_precomp"]
            opacity = ply_data["opacity"]
            shs = ply_data["shs"]
            quats = ply_data["quats"]
            scales = ply_data["scales"]
            gs_type = ply_data["gs_type"]

        elif source_type == "id_map":
            shared_ply = source.get("shared_ply")
            id_map_path = source.get("id_map")
            object_id = source.get("object_id")
            if shared_ply is None or id_map_path is None or object_id is None:
                raise ValueError(
                    f"Object '{name}': source.type='id_map' requires "
                    "'shared_ply', 'id_map', and 'object_id'."
                )
            ply_data = _load_ply_cached(shared_ply)
            id_map = _load_id_map_cached(id_map_path)
            id_mask_np = id_map == object_id
            id_mask = torch.from_numpy(id_mask_np).to(device=ply_data["pos"].device)

            pos = ply_data["pos"][id_mask]
            cov = ply_data["cov3D_precomp"][id_mask]
            opacity = ply_data["opacity"][id_mask]
            shs = ply_data["shs"][id_mask]
            quats = ply_data["quats"][id_mask]
            scales = ply_data["scales"][id_mask]
            gs_type = ply_data["gs_type"]

        elif source_type == "sim_area":
            shared_ply = source.get("shared_ply")
            sim_area = source.get("sim_area")
            if shared_ply is None or sim_area is None:
                raise ValueError(
                    f"Object '{name}': source.type='sim_area' requires "
                    "'shared_ply' and 'sim_area'."
                )
            ply_data = _load_ply_cached(shared_ply)
            pos = ply_data["pos"].clone()
            cov = ply_data["cov3D_precomp"].clone()
            opacity = ply_data["opacity"].clone()
            shs = ply_data["shs"].clone()
            quats = ply_data["quats"].clone()
            scales = ply_data["scales"].clone()
            gs_type = ply_data["gs_type"]
        else:
            raise ValueError(f"Object '{name}': unknown source.type '{source_type}'.")

        # --- Unified preprocessing ---

        pos, cov = _apply_axis_permutation(pos, cov, axis_perm)

        obj_opacity_threshold = obj_cfg.get("opacity_threshold", opacity_threshold)
        op_mask = opacity[:, 0] > obj_opacity_threshold
        pos = pos[op_mask]
        cov = cov[op_mask]
        opacity = opacity[op_mask]
        shs = shs[op_mask]
        quats = quats[op_mask]
        scales = scales[op_mask]

        if mode == "simulate":
            pos = apply_rotations(pos, rotation_matrices)
            cov = apply_cov_rotations(cov, rotation_matrices)
        elif source_type == "sim_area":
            pos = apply_rotations(pos, rotation_matrices)

        if source_type == "sim_area":
            sim_area = source["sim_area"]
            sa_mask = _apply_sim_area_mask(pos, sim_area)
            pos = pos[sa_mask]
            cov = cov[sa_mask]
            opacity = opacity[sa_mask]
            shs = shs[sa_mask]
            quats = quats[sa_mask]
            scales = scales[sa_mask]

        position_offset = obj_cfg.get("position_offset")
        if position_offset is not None and mode == "simulate":
            pos = pos + torch.tensor(
                position_offset, device=pos.device, dtype=pos.dtype,
            )

        obj_material_overrides = obj_cfg.get("material", {})
        material = _merge_material(top_material, obj_material_overrides)

        collider = obj_cfg.get("collider")
        particle_filling = obj_cfg.get("particle_filling")

        print(f"    -> {name}: {pos.shape[0]} GS particles (gs_type={gs_type})")

        result.append(SceneObject(
            name=name,
            mode=mode,
            positions=pos,
            covariances=cov,
            opacities=opacity,
            shs=shs,
            quats=quats,
            scales=scales,
            material=material,
            position_offset=position_offset,
            particle_filling=particle_filling,
            collider=collider,
            gs_type=gs_type,
        ))

    return result
