"""Shared rigid collision-geometry builders for Newton backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import torch
import warp as wp

from physics_sim.backend.newton_common.transforms import (
    rotmat_to_wp_quat_xyzw,
    unpack_cov6_to_3x3,
)
from physics_sim.geometry.convex_hull import compute_convex_hull
from physics_sim.geometry.primitives import fit_ellipsoid, fit_obb
from physics_sim.logging_utils import get_logger

try:
    from physics_sim.geometry.alpha_shape import compute_alpha_shape

    HAS_ALPHA_SHAPE = True
except ImportError:
    HAS_ALPHA_SHAPE = False


LOGGER = get_logger(__name__)
COLLISION_GEO_TYPES = ("obb", "ellipsoid", "convex_hull", "alpha_shape")


@dataclass
class BuiltRigidGeometry:
    body_idx: int
    init_local_pos: torch.Tensor
    init_cov_3x3: torch.Tensor
    init_quats: torch.Tensor | None
    init_scales: torch.Tensor | None


def normalize_collision_geo(
    raw_geo: str,
    *,
    log_prefix: str,
    logger: Any = LOGGER,
) -> str:
    """Validate and normalize collision geometry type."""
    if raw_geo not in COLLISION_GEO_TYPES:
        raise ValueError(
            f"Unknown collision_geometry={raw_geo!r}. Must be one of {COLLISION_GEO_TYPES}"
        )
    if raw_geo == "alpha_shape" and not HAS_ALPHA_SHAPE:
        logger.warning("[%s] open3d unavailable, fallback convex_hull", log_prefix)
        return "convex_hull"
    return raw_geo


def create_rigid_body_geometry(
    *,
    builder: newton.ModelBuilder,
    init_positions: torch.Tensor,
    init_covariances: torch.Tensor,
    init_quats: torch.Tensor | None,
    init_scales: torch.Tensor | None,
    device: str,
    particle_indices: list[int],
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    name: str,
    collision_geo: str | None,
    default_collision_geo: str,
    alpha: float | None,
    max_triangles: int,
    log_prefix: str,
    logger: Any = LOGGER,
) -> BuiltRigidGeometry:
    """Build one rigid body from selected particles and return back-mapping state."""
    idx_t = torch.tensor(particle_indices, dtype=torch.long)
    positions = init_positions[idx_t].float()
    pos_np = positions.detach().cpu().numpy()
    geo = normalize_collision_geo(
        collision_geo or default_collision_geo,
        log_prefix=log_prefix,
        logger=logger,
    )

    if geo in ("obb", "ellipsoid"):
        return _create_primitive_body(
            builder=builder,
            geo=geo,
            pos_np=pos_np,
            idx_t=idx_t,
            positions=positions,
            init_covariances=init_covariances,
            init_quats=init_quats,
            init_scales=init_scales,
            shape_cfg=shape_cfg,
            name=name,
            device=device,
            log_prefix=log_prefix,
            logger=logger,
        )
    return _create_mesh_body(
        builder=builder,
        geo=geo,
        pos_np=pos_np,
        idx_t=idx_t,
        positions=positions,
        init_covariances=init_covariances,
        init_quats=init_quats,
        init_scales=init_scales,
        shape_cfg=shape_cfg,
        name=name,
        alpha=alpha,
        max_triangles=max_triangles,
        device=device,
        log_prefix=log_prefix,
        logger=logger,
    )


def _create_primitive_body(
    *,
    builder: newton.ModelBuilder,
    geo: str,
    pos_np: np.ndarray,
    idx_t: torch.Tensor,
    positions: torch.Tensor,
    init_covariances: torch.Tensor,
    init_quats: torch.Tensor | None,
    init_scales: torch.Tensor | None,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    name: str,
    device: str,
    log_prefix: str,
    logger: Any,
) -> BuiltRigidGeometry:
    if geo == "obb":
        info = fit_obb(pos_np)
        hx, hy, hz = (float(v) for v in info["half_extents"])
    else:
        info = fit_ellipsoid(pos_np)

    center_np = info["center"]
    axes = info["axes"]
    rot = axes.T
    quat = rotmat_to_wp_quat_xyzw(rot)

    body_idx = builder.add_body(
        xform=wp.transform(p=wp.vec3(*center_np.astype(float)), q=wp.quat(*quat)),
        label=name,
    )

    if geo == "obb":
        builder.add_shape_box(body_idx, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
        shape_desc = f"box({hx:.3f}, {hy:.3f}, {hz:.3f})"
    else:
        sa = info["semi_axes"]
        a, b, c = (float(v) for v in sa)
        builder.add_shape_ellipsoid(body_idx, rx=a, ry=b, rz=c, cfg=shape_cfg)
        shape_desc = f"ellipsoid({a:.3f}, {b:.3f}, {c:.3f})"
    logger.info("[%s] body '%s': %s", log_prefix, name, shape_desc)

    rot_inv = np.linalg.inv(rot).astype(np.float32)
    local_pos_np = (pos_np - center_np) @ rot_inv.T
    local_pos = torch.from_numpy(local_pos_np).to(device)
    cov_3x3 = unpack_cov6_to_3x3(init_covariances[idx_t])
    iq = init_quats[idx_t] if init_quats is not None else None
    isc = init_scales[idx_t] if init_scales is not None else None
    return BuiltRigidGeometry(
        body_idx=body_idx,
        init_local_pos=local_pos,
        init_cov_3x3=cov_3x3,
        init_quats=iq,
        init_scales=isc,
    )


def _create_mesh_body(
    *,
    builder: newton.ModelBuilder,
    geo: str,
    pos_np: np.ndarray,
    idx_t: torch.Tensor,
    positions: torch.Tensor,
    init_covariances: torch.Tensor,
    init_quats: torch.Tensor | None,
    init_scales: torch.Tensor | None,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    name: str,
    alpha: float | None,
    max_triangles: int,
    device: str,
    log_prefix: str,
    logger: Any,
) -> BuiltRigidGeometry:
    center = positions.mean(dim=0)
    center_np = center.detach().cpu().numpy()

    if geo == "alpha_shape":
        try:
            mesh_verts, mesh_faces = compute_alpha_shape(
                pos_np,
                alpha=alpha,
                max_triangles=max_triangles,
            )
        except Exception as exc:
            logger.warning(
                "[%s] alpha shape failed for '%s': %s; fallback convex_hull",
                log_prefix,
                name,
                exc,
            )
            geo = "convex_hull"

    if geo == "convex_hull":
        try:
            mesh_verts, mesh_faces = compute_convex_hull(pos_np)
        except Exception as exc:
            logger.warning(
                "[%s] convex hull failed for '%s': %s; fallback bbox mesh",
                log_prefix,
                name,
                exc,
            )
            mesh_verts, mesh_faces = bbox_mesh(pos_np)

    local_verts = mesh_verts - center_np
    collision_mesh = newton.Mesh(
        local_verts.astype(np.float32),
        mesh_faces.flatten().astype(np.int32),
    )
    body_idx = builder.add_body(
        xform=wp.transform(p=wp.vec3(*center_np.astype(float)), q=wp.quat_identity()),
        label=name,
    )
    builder.add_shape_mesh(body_idx, mesh=collision_mesh, cfg=shape_cfg)

    cov_3x3 = unpack_cov6_to_3x3(init_covariances[idx_t])
    iq = init_quats[idx_t] if init_quats is not None else None
    isc = init_scales[idx_t] if init_scales is not None else None
    logger.info(
        "[%s] body '%s': mesh(%s)=%sv/%sf",
        log_prefix,
        name,
        geo,
        mesh_verts.shape[0],
        mesh_faces.shape[0],
    )
    return BuiltRigidGeometry(
        body_idx=body_idx,
        init_local_pos=(positions - center).to(device),
        init_cov_3x3=cov_3x3,
        init_quats=iq,
        init_scales=isc,
    )


def bbox_mesh(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fallback axis-aligned bounding box mesh with outward normals."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    extent = np.maximum(hi - lo, 1e-3)
    lo = lo - 0.01 * extent
    hi = hi + 0.01 * extent
    verts = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    center = (lo + hi) / 2.0
    for i in range(faces.shape[0]):
        v0, v1, v2 = verts[faces[i]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_center = (v0 + v1 + v2) / 3.0
        if np.dot(face_normal, face_center - center) < 0:
            faces[i, 1], faces[i, 2] = faces[i, 2], faces[i, 1]
    return verts, faces

