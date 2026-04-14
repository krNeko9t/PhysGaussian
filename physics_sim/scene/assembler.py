"""Scene assembler: reads config + PLY files and produces list[SceneObject].

Consumes the new Pydantic-based :class:`~physics_sim.config.models.SimConfig`
where each object is a typed ``ObjectConfig`` (not a raw dict).
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
from physics_sim.scene.objects import SceneObject


# ── Internal helpers ─────────────────────────────────────────────────

def _apply_axis_permutation(
    pos: torch.Tensor, cov: torch.Tensor, perm: str,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _resolve_path(path: str, config_dir: str | None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if config_dir is not None:
        candidate = Path(config_dir) / p
        if candidate.exists():
            return str(candidate)
    return str(p)


# ── Public API ───────────────────────────────────────────────────────

def assemble_scene(
    cfg: SimConfig,
    renderer: Any,
    config_dir: str | None = None,
) -> list[SceneObject]:
    """Build a list of :class:`SceneObject` from a :class:`SimConfig`.

    Args:
        cfg: Pydantic experiment config (fully typed).
        renderer: A ``GaussianRenderer`` instance (provides ``load_ply``).
        config_dir: Base directory for resolving relative PLY paths.

    Returns:
        One ``SceneObject`` per declared object, preprocessed and ready
        for consumption by the pipeline stages.
    """
    from physics_sim.preprocessing.transform import (
        apply_cov_rotations,
        apply_rotations,
        generate_rotation_matrices,
    )

    pp = cfg.preprocess
    axis_perm = pp.axis_permutation
    global_opacity_threshold = pp.opacity_threshold

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(pp.rotation_degree), pp.rotation_axis,
    )

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

        # ── Load GS data from source ─────────────────────────────────
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

        # ── Unified preprocessing ────────────────────────────────────
        pos, cov = _apply_axis_permutation(pos, cov, axis_perm)

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

        if role == "dynamic":
            pos = apply_rotations(pos, rotation_matrices)
            cov = apply_cov_rotations(cov, rotation_matrices)

        # ── Transform (position offset + future rotation) ────────────
        transform = obj_cfg.transform
        offset = transform.position
        if role == "dynamic" and any(v != 0.0 for v in offset):
            pos = pos + torch.tensor(offset, device=pos.device, dtype=pos.dtype)

        # ── Collider / filling as dicts for downstream ───────────────
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
