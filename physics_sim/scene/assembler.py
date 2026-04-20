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

import numpy as np
import torch
from plyfile import PlyData

from physics_sim.config.models import (
    IdMapSource,
    PointSelectorSource,
    PlySource,
    SimConfig,
)
from physics_sim.render.interfaces import SceneAssetLoader
from physics_sim.render.types import GaussianAsset
from physics_sim.coord import (
    SourceAxes,
    align_covariances,
    align_positions,
    align_quats,
)
from physics_sim.logging_utils import get_logger
from physics_sim.scene.objects import SceneObject

LOGGER = get_logger(__name__)


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


def _load_xyz_points_from_ply(path: str) -> np.ndarray:
    ply_data = PlyData.read(path)
    if not ply_data.elements:
        raise ValueError(f"PLY has no elements: {path}")
    vtx = ply_data.elements[0]
    names = {p.name for p in vtx.properties}
    required = {"x", "y", "z"}
    missing = required - names
    if missing:
        miss = ", ".join(sorted(missing))
        raise ValueError(f"PLY missing xyz properties ({miss}): {path}")
    xyz = np.stack(
        [np.asarray(vtx["x"]), np.asarray(vtx["y"]), np.asarray(vtx["z"])],
        axis=1,
    ).astype(np.float64, copy=False)
    return xyz


def _load_index_array(path: str) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    if isinstance(raw, np.lib.npyio.NpzFile):
        keys = list(raw.files)
        if len(keys) != 1:
            joined = ", ".join(keys)
            raise ValueError(
                "Index npz must contain exactly one array, "
                f"got {len(keys)} keys: {joined}",
            )
        arr = raw[keys[0]]
    else:
        arr = raw
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Index list must be 1D, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Index list dtype must be integer, got {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _quantize_xyz(xyz: np.ndarray, tol: float) -> np.ndarray:
    # Match by quantized coordinates to tolerate tiny float serialization drift.
    return np.rint(xyz / tol).astype(np.int64)


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

    if not cfg.objects:
        raise ValueError("Config must declare at least one object.")

    ply_cache: dict[str, GaussianAsset] = {}
    id_map_cache: dict[str, np.ndarray] = {}
    selector_point_cache: dict[str, np.ndarray] = {}
    selector_index_cache: dict[str, np.ndarray] = {}
    base_lookup_cache: dict[tuple[str, float], tuple[np.ndarray, dict[tuple[int, int, int], list[int]]]] = {}

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

    def _load_selector_points(path: str, object_name: str) -> np.ndarray:
        resolved = _resolve_for_object(path, object_name=object_name, source_type="point_selector")
        if resolved not in selector_point_cache:
            try:
                selector_point_cache[resolved] = _load_xyz_points_from_ply(resolved)
            except Exception as exc:
                raise ValueError(
                    f"{_ctx(object_name, 'point_selector')} failed to load point selector "
                    f"from {resolved}: {exc}",
                ) from exc
        return selector_point_cache[resolved]

    def _load_selector_indices(path: str, object_name: str) -> np.ndarray:
        resolved = _resolve_for_object(path, object_name=object_name, source_type="point_selector")
        if resolved not in selector_index_cache:
            try:
                selector_index_cache[resolved] = _load_index_array(resolved)
            except Exception as exc:
                raise ValueError(
                    f"{_ctx(object_name, 'point_selector')} failed to load index selector "
                    f"from {resolved}: {exc}",
                ) from exc
        return selector_index_cache[resolved]

    def _selector_to_index_mask(
        *,
        object_name: str,
        selector_kind: str,
        selector_path: str,
        base_path: str,
        base_positions: np.ndarray,
        tolerance: float,
        strict: bool,
        min_match_ratio: float,
        max_ambiguous_ratio: float,
    ) -> np.ndarray:
        if selector_kind == "index_list":
            idx = _load_selector_indices(selector_path, object_name=object_name)
            invalid = (idx < 0) | (idx >= base_positions.shape[0])
            invalid_count = int(invalid.sum())
            if invalid_count > 0:
                msg = (
                    f"{_ctx(object_name, 'point_selector')} invalid index_list values: "
                    f"invalid={invalid_count}, total={idx.shape[0]}, "
                    f"base_gs={base_positions.shape[0]}, selector={selector_path}"
                )
                if strict:
                    raise ValueError(msg)
                LOGGER.warning(msg + " ; dropping invalid entries")
                idx = idx[~invalid]
            unique_idx = np.unique(idx)
            ratio = 0.0 if idx.shape[0] == 0 else float(unique_idx.shape[0] / idx.shape[0])
            if strict and ratio < min_match_ratio:
                raise ValueError(
                    f"{_ctx(object_name, 'point_selector')} low effective index match ratio: "
                    f"ratio={ratio:.6f} < min_match_ratio={min_match_ratio:.6f}, "
                    f"selector={selector_path}",
                )
            if (not strict) and ratio < min_match_ratio:
                LOGGER.warning(
                    "%s low effective index match ratio ratio=%.6f min=%.6f selector=%s",
                    _ctx(object_name, "point_selector"),
                    ratio,
                    min_match_ratio,
                    selector_path,
                )
            mask = np.zeros(base_positions.shape[0], dtype=bool)
            mask[unique_idx] = True
            LOGGER.info(
                "%s selector=index_list selector_count=%s matched_gs=%s selector=%s",
                _ctx(object_name, "point_selector"),
                idx.shape[0],
                unique_idx.shape[0],
                selector_path,
            )
            return mask

        if selector_kind != "point_cloud_xyz":
            raise ValueError(
                f"{_ctx(object_name, 'point_selector')} unsupported selector_kind={selector_kind}",
            )

        selector_points = _load_selector_points(selector_path, object_name=object_name)
        if selector_points.shape[0] == 0:
            raise ValueError(
                f"{_ctx(object_name, 'point_selector')} empty selector point cloud: {selector_path}",
            )
        if selector_points.shape[1] != 3:
            raise ValueError(
                f"{_ctx(object_name, 'point_selector')} selector points must be Nx3, "
                f"got shape={selector_points.shape}, selector={selector_path}",
            )

        cache_key = (base_path, tolerance)
        if cache_key not in base_lookup_cache:
            base_keys = _quantize_xyz(base_positions, tolerance)
            lookup: dict[tuple[int, int, int], list[int]] = {}
            for i, key in enumerate(base_keys):
                tup = (int(key[0]), int(key[1]), int(key[2]))
                if tup in lookup:
                    lookup[tup].append(i)
                else:
                    lookup[tup] = [i]
            base_lookup_cache[cache_key] = (base_keys, lookup)
        _, lookup = base_lookup_cache[cache_key]

        selector_keys = _quantize_xyz(selector_points, tolerance)
        matched_indices: list[int] = []
        unmatched_count = 0
        ambiguous_count = 0
        for i, sk in enumerate(selector_keys):
            candidates = lookup.get((int(sk[0]), int(sk[1]), int(sk[2])))
            if not candidates:
                unmatched_count += 1
                continue
            if len(candidates) == 1:
                matched_indices.append(candidates[0])
                continue

            ambiguous_count += 1
            base_subset = base_positions[np.asarray(candidates, dtype=np.int64)]
            d2 = np.sum((base_subset - selector_points[i]) ** 2, axis=1)
            best = int(np.argmin(d2))
            matched_indices.append(candidates[best])

        total = int(selector_points.shape[0])
        matched = int(len(matched_indices))
        ratio = 0.0 if total == 0 else float(matched / total)
        unmatched_ratio = 0.0 if total == 0 else float(unmatched_count / total)
        ambiguous_ratio = 0.0 if total == 0 else float(ambiguous_count / total)

        msg_prefix = (
            f"{_ctx(object_name, 'point_selector')} selector={selector_path} "
            f"selector_points={total} matched={matched} unmatched={unmatched_count} "
            f"ambiguous={ambiguous_count} tolerance={tolerance}"
        )
        if strict and ambiguous_ratio > max_ambiguous_ratio:
            raise ValueError(
                msg_prefix
                + " ; ambiguous_ratio="
                + f"{ambiguous_ratio:.6f} > max_ambiguous_ratio={max_ambiguous_ratio:.6f}",
            )
        if strict and ratio < min_match_ratio:
            raise ValueError(
                msg_prefix
                + f" ; match_ratio={ratio:.6f} < min_match_ratio={min_match_ratio:.6f}",
            )
        if (not strict) and ambiguous_ratio > max_ambiguous_ratio:
            LOGGER.warning(
                "%s high ambiguous ratio ambiguous_ratio=%.6f max=%.6f",
                msg_prefix,
                ambiguous_ratio,
                max_ambiguous_ratio,
            )
        if (not strict) and ratio < min_match_ratio:
            LOGGER.warning(
                "%s low match ratio ratio=%.6f min=%.6f",
                msg_prefix,
                ratio,
                min_match_ratio,
            )

        if matched == 0:
            raise ValueError(msg_prefix + " ; selector produced zero matches")

        unique_idx = np.unique(np.asarray(matched_indices, dtype=np.int64))
        mask = np.zeros(base_positions.shape[0], dtype=bool)
        mask[unique_idx] = True
        LOGGER.info(
            "%s matched_gs=%s unique_ratio=%.6f unmatched_ratio=%.6f",
            msg_prefix,
            unique_idx.shape[0],
            0.0 if matched == 0 else float(unique_idx.shape[0] / matched),
            unmatched_ratio,
        )
        return mask

    result: list[SceneObject] = []

    for obj_cfg in cfg.objects:
        name = obj_cfg.name
        role = obj_cfg.role
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
            include_mask_np = _selector_to_index_mask(
                object_name=name,
                selector_kind=source.selector_kind,
                selector_path=source.selector_path,
                base_path=base_resolved,
                base_positions=base_positions,
                tolerance=source.match_tolerance,
                strict=source.strict,
                min_match_ratio=source.min_match_ratio,
                max_ambiguous_ratio=source.max_ambiguous_ratio,
            )
            final_mask_np = include_mask_np
            if source.invert:
                final_mask_np = ~final_mask_np
            if source.subtract_selector_path is not None:
                subtract_kind = source.subtract_selector_kind or source.selector_kind
                subtract_mask_np = _selector_to_index_mask(
                    object_name=name,
                    selector_kind=subtract_kind,
                    selector_path=source.subtract_selector_path,
                    base_path=base_resolved,
                    base_positions=base_positions,
                    tolerance=source.match_tolerance,
                    strict=source.strict,
                    min_match_ratio=source.min_match_ratio,
                    max_ambiguous_ratio=source.max_ambiguous_ratio,
                )
                final_mask_np = final_mask_np & (~subtract_mask_np)

            selected_count = int(final_mask_np.sum())
            if selected_count == 0:
                raise ValueError(
                    f"{_ctx(name, source.type)} selector produced empty GS subset after "
                    f"invert/subtract, base_gs={ply_data.pos.shape[0]}",
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
