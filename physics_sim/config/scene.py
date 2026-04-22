"""Scene-graph configuration: parts + constraints.

Replaces the flat ``SimConfig.objects`` list with an explicit structure:

- ``parts``       — each part has its own geometry, transform, material.
  (Essentially the old ``ObjectConfig`` without the ``role`` field.)
- ``constraints`` — typed relations between parts, or between a part and
  the world.  Backends translate them to concrete operations (particle
  mass overrides, setup_collider calls, per-step kernels).

The v1 constraint set is intentionally small:

- ``PinToWorld``  — freeze a set of particles in their spawn positions.
- ``PinToBody``   — freeze a set of particles in a rigid body's local frame.
- ``CollideOnly`` — part participates only as a collider (VBD: contact
  pipeline; MPM: ``setup_collider``).

See ``docs`` / plan file for semantics and backend translation.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

from physics_sim.config.models import (
    ColliderConfig,
    FillingConfig,
    MaterialSpec,
    ObjectSource,
    ObjectTransform,
)


# ── Part ────────────────────────────────────────────────────────────

class PartConfig(BaseModel):
    """Single-part description (geometry + material).

    Successor to ``ObjectConfig``; no ``role`` field — collision/render
    behavior is expressed by which constraints reference the part.
    A ``material=None`` part is treated as render-only.
    """
    name: str
    source: ObjectSource
    transform: ObjectTransform = Field(default_factory=ObjectTransform)
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material: MaterialSpec | None = None
    particle_filling: FillingConfig | None = None
    # Parts sharing the same fill_group are unioned before filling so the
    # interior of a logically-single body isn't torn apart by seams.
    # None = the part stands alone (still filled if particle_filling is set).
    fill_group: str | None = None
    collider: ColliderConfig | None = None
    opacity_threshold: float | None = None


# ── ParticleSelector ────────────────────────────────────────────────

class ProximitySelector(BaseModel):
    """Pick particles of ``part`` within ``max_distance`` of another part's surface."""
    kind: Literal["proximity"] = "proximity"
    part: str
    to_surface_of: str
    max_distance: float = Field(gt=0.0)


class BoxSelector(BaseModel):
    """Pick particles of ``part`` inside an axis-aligned box."""
    kind: Literal["box"] = "box"
    part: str
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    space: Literal["world", "part_local"] = "world"

    @model_validator(mode="after")
    def _min_le_max(self) -> "BoxSelector":
        if any(mn > mx for mn, mx in zip(self.min, self.max)):
            raise ValueError(f"BoxSelector min {self.min} must be <= max {self.max}")
        return self


class IndexSelector(BaseModel):
    """Pick explicit part-local particle indices (or a path to a .npy file)."""
    kind: Literal["index"] = "index"
    part: str
    indices: list[int] | str


ParticleSelector = Annotated[
    Union[ProximitySelector, BoxSelector, IndexSelector],
    Field(discriminator="kind"),
]


def _selector_part(selector) -> str:
    return selector.part


def _selector_referenced_parts(selector) -> list[str]:
    """All parts a selector references (for validation)."""
    refs = [selector.part]
    if isinstance(selector, ProximitySelector):
        refs.append(selector.to_surface_of)
    return refs


# ── Constraint ──────────────────────────────────────────────────────

class PinToWorld(BaseModel):
    """Freeze selected particles at their spawn positions (mass -> 0)."""
    kind: Literal["pin_to_world"] = "pin_to_world"
    particles: ParticleSelector


class PinToBody(BaseModel):
    """Selected particles follow a rigid body's transform (mass -> 0 + per-step overwrite)."""
    kind: Literal["pin_to_body"] = "pin_to_body"
    particles: ParticleSelector
    body: str  # name of a rigid-material part


class CollideOnly(BaseModel):
    """A part participates only as a collider, not as a force-bearing body."""
    kind: Literal["collide_only"] = "collide_only"
    part: str


ConstraintConfig = Annotated[
    Union[PinToWorld, PinToBody, CollideOnly],
    Field(discriminator="kind"),
]


def _constraint_referenced_parts(c) -> list[str]:
    """All part names a constraint mentions (for cross-validation)."""
    if isinstance(c, PinToWorld):
        return _selector_referenced_parts(c.particles)
    if isinstance(c, PinToBody):
        return _selector_referenced_parts(c.particles) + [c.body]
    if isinstance(c, CollideOnly):
        return [c.part]
    raise TypeError(f"Unknown constraint type: {type(c).__name__}")


# ── Scene ───────────────────────────────────────────────────────────

class SceneConfig(BaseModel):
    """Ordered list of parts plus a list of constraints between them."""
    parts: list[PartConfig]
    constraints: list[ConstraintConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_part_names_unique(self) -> "SceneConfig":
        names = [p.name for p in self.parts]
        dups = {n for n in names if names.count(n) > 1}
        if dups:
            raise ValueError(f"Duplicate part names in scene: {sorted(dups)}")
        return self

    @model_validator(mode="after")
    def _check_constraint_references(self) -> "SceneConfig":
        names = {p.name for p in self.parts}
        for i, c in enumerate(self.constraints):
            for ref in _constraint_referenced_parts(c):
                if ref not in names:
                    raise ValueError(
                        f"constraint[{i}] ({type(c).__name__}) references unknown part '{ref}'"
                    )
        return self

    @model_validator(mode="after")
    def _check_pin_to_body_target_is_rigid(self) -> "SceneConfig":
        # Import locally to avoid exposing RigidMaterial in this module's top-level deps.
        from physics_sim.config.models import RigidMaterial, VBDMaterial, VBDRigidBody

        parts_by_name = {p.name: p for p in self.parts}
        for i, c in enumerate(self.constraints):
            if not isinstance(c, PinToBody):
                continue
            target = parts_by_name[c.body]
            mat = target.material
            if mat is None:
                raise ValueError(
                    f"constraint[{i}] PinToBody.body='{c.body}' has no material; "
                    "target must be a rigid part."
                )
            is_rigid = isinstance(mat, RigidMaterial) or (
                isinstance(mat, VBDMaterial) and isinstance(mat.body, VBDRigidBody)
            )
            if not is_rigid:
                raise ValueError(
                    f"constraint[{i}] PinToBody.body='{c.body}' must be a rigid part; "
                    f"got material type {type(mat).__name__}."
                )
        return self

    @model_validator(mode="after")
    def _check_fill_group_consistency(self) -> "SceneConfig":
        groups: dict[str, list[PartConfig]] = {}
        for p in self.parts:
            if p.fill_group is None:
                continue
            groups.setdefault(p.fill_group, []).append(p)
        for name, members in groups.items():
            filled = [m for m in members if m.particle_filling is not None]
            if not filled:
                continue
            ref_cfg = filled[0].particle_filling
            for m in filled[1:]:
                if m.particle_filling != ref_cfg:
                    raise ValueError(
                        f"fill_group '{name}': parts must share the same FillingConfig; "
                        f"part '{m.name}' differs from '{filled[0].name}'"
                    )
        return self


# Resolve the forward reference in models.SimConfig now that SceneConfig exists.
from physics_sim.config import models as _models  # noqa: E402

_models.SimConfig.model_rebuild(_types_namespace={"SceneConfig": SceneConfig})
