"""Typed configuration schema for the simulation pipeline.

Three fixed-schema sections (time, preprocess, camera) are modelled as
dataclasses so that defaults live in one place and the IDE can autocomplete
field names.  Open-ended sections (backend, material) remain plain dicts
because the set of valid keys varies per backend/material type.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Optional


def _from_dict(cls, d: dict):
    """Construct a dataclass from *d*, silently dropping unknown keys."""
    known = {f.name for f in fields(cls)}
    extra = set(d) - known
    if extra:
        warnings.warn(
            f"[{cls.__name__}] ignoring unknown fields: {extra}",
            stacklevel=2,
        )
    return cls(**{k: v for k, v in d.items() if k in known})


# -- Fixed-schema sections -------------------------------------------------

@dataclass
class TimeConfig:
    substep_dt: float = 1e-4
    frame_dt: float = 1e-2
    frame_num: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> TimeConfig:
        return _from_dict(cls, d)


@dataclass
class PreprocessConfig:
    opacity_threshold: float = 0.02
    axis_permutation: str = "xyz"
    rotation_degree: list[float] = field(default_factory=lambda: [0.0])
    rotation_axis: list[int] = field(default_factory=lambda: [0])
    scale: float = 1.0
    transform_reference: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessConfig:
        return _from_dict(cls, d)


@dataclass
class CameraConfig:
    camera_mode: str = "orbit"
    cameras_json: Optional[str] = None
    mpm_space_viewpoint_center: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    mpm_space_vertical_upward_axis: list[float] = field(
        default_factory=lambda: [0, 0, 1]
    )
    default_camera_index: int = -1
    show_hint: bool = False
    init_azimuth: float = 170.0
    init_elevation: float = 0.0
    init_radius: float = 1.6
    move_camera: bool = True
    delta_a: float = -2.4
    delta_e: float = 0.8
    delta_r: float = 0.0
    width: Optional[int] = None
    height: Optional[int] = None
    fovx_deg: Optional[float] = None
    fovy_deg: Optional[float] = None
    fx: Optional[float] = None
    fy: Optional[float] = None
    fixed_position: Optional[list[float]] = None
    fixed_rotation: Optional[list[float]] = None

    @classmethod
    def from_dict(cls, d: dict) -> CameraConfig:
        return _from_dict(cls, d)


# -- Top-level config container --------------------------------------------

@dataclass
class SimConfig:
    """Fully resolved simulation configuration.

    Constructed by :class:`ConfigLoader` after YAML loading and ``_ref``
    resolution.  All fragment references have already been expanded.
    """

    output: str
    backend_type: str
    backend: dict[str, Any]
    material: dict[str, Any]
    time: TimeConfig
    preprocess: PreprocessConfig
    camera: CameraConfig
    objects: list[dict[str, Any]]
    boundary_conditions: list[dict[str, Any]] = field(default_factory=list)
    particle_filling: Optional[dict[str, Any]] = None
