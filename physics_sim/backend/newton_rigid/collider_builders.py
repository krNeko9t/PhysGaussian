"""Rigid collider builders for NewtonRigid backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
import newton

from physics_sim.backend.newton_common import (
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
class BuiltRigidBody:
    body_idx: int
    init_local_pos: torch.Tensor
    init_cov_3x3: torch.Tensor
    init_quats: torch.Tensor | None
    init_scales: torch.Tensor | None


def normalize_collision_geo(raw_geo: str) -> str:
    """Validate and normalize collision geometry type."""
    if raw_geo not in COLLISION_GEO_TYPES:
        raise ValueError(
            f"Unknown collision_geometry={raw_geo!r}. Must be one of {COLLISION_GEO_TYPES}"
        )
    if raw_geo == "alpha_shape" and not HAS_ALPHA_SHAPE:
        LOGGER.warning(
            "[NewtonRigid] open3d unavailable, fallback convex_hull"
        )
        return "convex_hull"
    return raw_geo


def create_rigid_body(
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
    collision_geo: str,
    alpha: float | None,
    max_triangles: int,
) -> BuiltRigidBody:
    idx_t = torch.tensor(particle_indices, dtype=torch.long)
    positions = init_positions[idx_t]
    pos_np = positions.detach().cpu().numpy()
    if collision_geo in ("obb", "ellipsoid"):
        return _create_primitive_body(
            builder=builder,
            geo=collision_geo,
            pos_np=pos_np,
            idx_t=idx_t,
            positions=positions,
            init_covariances=init_covariances,
            init_quats=init_quats,
            init_scales=init_scales,
            shape_cfg=shape_cfg,
            name=name,
            device=device,
        )
    return _create_mesh_body(
        builder=builder,
        geo=collision_geo,
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
) -> BuiltRigidBody:
    if geo == "obb":
        info = fit_obb(pos_np)
        hx, hy, hz = (float(v) for v in info["half_extents"])
    else:
        info = fit_ellipsoid(pos_np)

    center_np = info["center"]
    axes = info["axes"]
    R = axes.T
    quat = rotmat_to_wp_quat_xyzw(R)

    body_idx = builder.add_body(
        xform=wp.transform(
            p=wp.vec3(float(center_np[0]), float(center_np[1]), float(center_np[2])),
            q=wp.quat(*quat),
        ),
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
    LOGGER.info("[NewtonRigid] body '%s': %s", name, shape_desc)

    center_t = torch.tensor(center_np, device=device, dtype=torch.float32)
    R_t = torch.tensor(R, device=device, dtype=torch.float32)
    local_pos = ((positions.float() - center_t) @ R_t).to(device)
    cov_3x3 = unpack_cov6_to_3x3(init_covariances[idx_t])
    iq = init_quats[idx_t] if init_quats is not None else None
    isc = init_scales[idx_t] if init_scales is not None else None
    return BuiltRigidBody(
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
) -> BuiltRigidBody:
    center = positions.mean(dim=0)
    if geo == "alpha_shape":
        try:
            mesh_verts, mesh_faces = compute_alpha_shape(
                pos_np,
                alpha=alpha,
                max_triangles=max_triangles,
            )
        except Exception as exc:
            LOGGER.warning(
                "[NewtonRigid] alpha shape failed for '%s': %s; fallback convex_hull",
                name,
                exc,
            )
            geo = "convex_hull"

    if geo == "convex_hull":
        try:
            mesh_verts, mesh_faces = compute_convex_hull(pos_np)
        except Exception as exc:
            LOGGER.warning(
                "[NewtonRigid] convex hull failed for '%s': %s; fallback bbox mesh",
                name,
                exc,
            )
            mesh_verts, mesh_faces = bbox_mesh(pos_np)

    center_np = center.detach().cpu().numpy()
    local_verts = mesh_verts - center_np
    collision_mesh = newton.Mesh(
        local_verts.astype(np.float32),
        mesh_faces.flatten().astype(np.int32),
    )
    body_idx = builder.add_body(
        xform=wp.transform(
            p=wp.vec3(float(center_np[0]), float(center_np[1]), float(center_np[2])),
            q=wp.quat_identity(),
        ),
        label=name,
    )
    builder.add_shape_mesh(body_idx, mesh=collision_mesh, cfg=shape_cfg)

    cov_3x3 = unpack_cov6_to_3x3(init_covariances[idx_t])
    iq = init_quats[idx_t] if init_quats is not None else None
    isc = init_scales[idx_t] if init_scales is not None else None
    LOGGER.info(
        "[NewtonRigid] body '%s': mesh(%s)=%sv/%sf",
        name,
        geo,
        mesh_verts.shape[0],
        mesh_faces.shape[0],
    )
    return BuiltRigidBody(
        body_idx=body_idx,
        init_local_pos=(positions - center).to(device),
        init_cov_3x3=cov_3x3,
        init_quats=iq,
        init_scales=isc,
    )


def bbox_mesh(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding-box mesh fallback."""
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
