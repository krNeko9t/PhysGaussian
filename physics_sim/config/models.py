"""Pydantic configuration models for the simulation pipeline.

All experiment configuration is expressed as typed Pydantic objects.
Experiment files are Python scripts that construct a ``SimConfig``
directly -- no YAML parsing, no ``_ref`` resolution, no deep-merge.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ── Data sources ─────────────────────────────────────────────────────

class PlySource(BaseModel):
    type: Literal["ply"] = "ply"
    ply_path: str


class IdMapSource(BaseModel):
    type: Literal["id_map"] = "id_map"
    ply_path: str
    id_map: str
    object_id: int


class PointSelectorSource(BaseModel):
    """Select GS subset from a base PLY via external selector data.

    selector_kind contract:
    - ``point_cloud_xyz``: selector_path points to a PLY containing vertex x/y/z.
    - ``index_list``: selector_path points to a numpy array of integer indices.
    """

    type: Literal["point_selector"] = "point_selector"
    base_ply_path: str
    selector_path: str
    selector_kind: Literal["point_cloud_xyz", "index_list"]
    invert: bool = False
    subtract_selector_path: str | None = None
    subtract_selector_kind: Literal["point_cloud_xyz", "index_list"] | None = None
    match_tolerance: float = Field(default=1e-6, gt=0.0)
    min_match_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    max_ambiguous_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    strict: bool = True


ObjectSource = Annotated[
    Union[PlySource, IdMapSource, PointSelectorSource],
    Field(discriminator="type"),
]


# ── Object transform ────────────────────────────────────────────────

class ObjectTransform(BaseModel):
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ── Collider ─────────────────────────────────────────────────────────

class ColliderConfig(BaseModel):
    type: str = "plane"
    space: str = "world"
    surface: str = "slip"
    friction: float = 0.0
    point: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] | None = None
    prefer_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    render: bool = True
    fit: dict | None = None


# ── Particle filling ────────────────────────────────────────────────

class FillingConfig(BaseModel):
    n_grid: int = 128
    density_threshold: float = 20.0
    search_threshold: float = 0.2
    search_exclude_direction: int = -1
    ray_cast_direction: int = 4
    max_particles_num: int = 500_000
    max_particles_per_cell: int = 4
    boundary: list[float] = Field(
        default_factory=lambda: [0.2, 1.8, 0.2, 1.8, 0.2, 1.8],
    )
    smooth: bool = True
    visualize: bool = False


# ── Object ───────────────────────────────────────────────────────────

class ObjectConfig(BaseModel):
    name: str
    source: ObjectSource
    role: Literal["dynamic", "collider_only", "render_only"] = "dynamic"
    transform: ObjectTransform = Field(default_factory=ObjectTransform)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: dict = Field(default_factory=dict)
    particle_filling: FillingConfig | None = None
    collider: ColliderConfig | None = None
    opacity_threshold: float | None = None


# ── Backend configs (one typed class per backend) ────────────────────

class NewtonRigidConfig(BaseModel):
    type: Literal["newton_rigid"] = "newton_rigid"
    n_grid: int = 200
    grid_lim: float = 2.0
    collision_geometry: str = "convex_hull"
    use_sdf: bool = False
    sdf_resolution: int = 64
    contact_margin: float | None = None
    solver_iterations: int = 10
    contact_relaxation: float = 0.8


class NewtonVBDConfig(BaseModel):
    type: Literal["newton_vbd"] = "newton_vbd"
    n_grid: int = 200
    grid_lim: float = 2.0
    collision_geometry: str = "convex_hull"
    use_sdf: bool = True
    contact_margin: float = 0.01
    soft_contact_ke: float | None = None
    soft_contact_kd: float | None = None
    soft_contact_mu: float | None = None
    debug_soft_no_deformation: bool = False
    sv_clamp_min: float | None = None
    sv_clamp_max: float | None = None
    particle_self_contact: bool | None = None
    particle_self_contact_radius: float | None = None
    particle_self_contact_margin: float | None = None
    rigid_contact_max: int | None = None
    solver_iterations: int = 20
    contact_relaxation: float = 0.5


class NewtonMPMConfig(BaseModel):
    type: Literal["newton_mpm"] = "newton_mpm"
    n_grid: int = 200
    grid_lim: float = 2.0
    solver_iterations: int | None = None
    solver_tolerance: float | None = None
    transfer_scheme: str | None = None


class NoneBackendConfig(BaseModel):
    type: Literal["none"] = "none"


BackendConfig = Annotated[
    Union[NewtonRigidConfig, NewtonVBDConfig, NewtonMPMConfig, NoneBackendConfig],
    Field(discriminator="type"),
]


# ── Time / Preprocess / Camera ───────────────────────────────────────

class TimeConfig(BaseModel):
    substep_dt: float = 1e-4
    frame_dt: float = 1e-2
    frame_num: int = 100


class PreprocessConfig(BaseModel):
    opacity_threshold: float = 0.02
    source_up: str = "+Y"
    source_front: str | None = None
    scale: float = 1.0


class CameraConfig(BaseModel):
    camera_mode: Literal["orbit", "fixed", "external"] = "orbit"
    camera_format: Literal[
        "colmap",
        "blender",
        "nerfstudio",
        "physgaussian",
        "opencv",
        "opengl",
    ] | None = None
    camera_path: str | None = None
    camera_pose_convention: Literal["opencv_w2c", "opengl_c2w"] | None = None
    camera_world_frame: Literal["source", "internal"] | None = None
    camera_index: int = Field(default=0, ge=0)
    init_azimuth: float = 170.0
    init_elevation: float = 0.0
    init_radius: float = 1.6
    move_camera: bool = True
    delta_a: float = -2.4
    delta_e: float = 0.8
    delta_r: float = 0.0
    # Procedural orbit/fixed cameras need a framebuffer size; external mode reads
    # intrinsics directly from the external camera payload.
    width: int = 800
    height: int = 600
    fovx_deg: float = 60
    fovy_deg: float = 45
    fx: float | None = None
    fy: float | None = None
    fixed_position: list[float] | None = None
    fixed_rotation: list[float] | None = None


# ── Boundary conditions ──────────────────────────────────────────────

class SurfaceCollider(BaseModel):
    type: Literal["surface_collider"] = "surface_collider"
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    surface: str = "slip"
    friction: float = 0.0
    space: str = "world"
    start_time: float = 0.0
    end_time: float = 1e3


class BoundingBox(BaseModel):
    type: Literal["bounding_box"] = "bounding_box"


class ReleaseParticlesSequentially(BaseModel):
    type: Literal["release_particles_sequentially"] = "release_particles_sequentially"
    normal: tuple[float, float, float]
    start_position: float
    end_position: float
    num_layers: int = 50
    start_time: float = 0.0
    end_time: float = 1.5


BoundaryCondition = Union[SurfaceCollider, BoundingBox, ReleaseParticlesSequentially]


# ── Top-level config ─────────────────────────────────────────────────

class SimConfig(BaseModel):
    """Complete experiment configuration.

    Constructed directly in a Python experiment file, e.g.::

        from physics_sim.config.models import *
        config = SimConfig(backend=NewtonRigidConfig(), objects=[...])
    """

    output: str = "output"
    backend: BackendConfig
    time: TimeConfig = Field(default_factory=TimeConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    objects: list[ObjectConfig]
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
