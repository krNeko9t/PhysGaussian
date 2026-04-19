"""Common Newton backend helpers."""

from .boundary import (
    build_bounding_box_planes,
    plane_from_point_normal,
    resolve_surface_friction,
    surface_plane_from_bc,
)
from .rigid_geometry import (
    COLLISION_GEO_TYPES,
    BuiltRigidGeometry,
    create_rigid_body_geometry,
    normalize_collision_geo,
)
from .rigid_state import populate_rigid_particles, sanitize_body_poses
from .transforms import (
    compose_body_quat_wxyz,
    pack_cov3x3_to_6,
    quat_xyzw_to_rotmat,
    rotmat_to_wp_quat_xyzw,
    unpack_cov6_to_3x3,
)

__all__ = [
    "BuiltRigidGeometry",
    "COLLISION_GEO_TYPES",
    "build_bounding_box_planes",
    "compose_body_quat_wxyz",
    "create_rigid_body_geometry",
    "normalize_collision_geo",
    "populate_rigid_particles",
    "pack_cov3x3_to_6",
    "plane_from_point_normal",
    "quat_xyzw_to_rotmat",
    "resolve_surface_friction",
    "rotmat_to_wp_quat_xyzw",
    "sanitize_body_poses",
    "surface_plane_from_bc",
    "unpack_cov6_to_3x3",
]
