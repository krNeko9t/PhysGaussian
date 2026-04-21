"""Pydantic-level tests for the new ``physics_sim.config.scene`` module.

These don't require torch/warp/GPU — they validate schema shape and
cross-field checks only.
"""

from __future__ import annotations

import sys
import types

import pytest


def _stub_torch() -> None:
    """Install a minimal torch stub so ``physics_sim.config`` imports cleanly.

    The config layer itself never touches torch, but importing the
    physics_sim package triggers torch imports deeper in the tree.
    """
    if "torch" in sys.modules:
        return
    mod = types.ModuleType("torch")
    mod.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = mod


_stub_torch()


from physics_sim.config.models import (  # noqa: E402
    NewtonVBDConfig,
    PlySource,
    RigidMaterial,
    SimConfig,
    VBDMaterial,
    VBDRigidBody,
    VBDSoftBody,
)
from physics_sim.config.scene import (  # noqa: E402
    BoxSelector,
    CollideOnly,
    IndexSelector,
    PartConfig,
    PinToBody,
    PinToWorld,
    ProximitySelector,
    SceneConfig,
)


# ── PartConfig / SceneConfig ───────────────────────────────────────

def test_part_config_basic():
    p = PartConfig(name="pot", source=PlySource(ply_path="x.ply"))
    assert p.name == "pot"
    assert p.material is None  # render_only by omission


def test_scene_duplicate_part_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        SceneConfig(parts=[
            PartConfig(name="x", source=PlySource(ply_path="a.ply")),
            PartConfig(name="x", source=PlySource(ply_path="b.ply")),
        ])


def test_scene_unknown_constraint_target_rejected():
    with pytest.raises(ValueError, match="unknown part"):
        SceneConfig(
            parts=[PartConfig(name="p", source=PlySource(ply_path="a.ply"))],
            constraints=[CollideOnly(part="nope")],
        )


def test_pin_to_body_requires_rigid_target():
    """Validator rejects PinToBody whose 'body' is a soft part."""
    with pytest.raises(ValueError, match="must be a rigid part"):
        SceneConfig(
            parts=[
                PartConfig(
                    name="soft",
                    source=PlySource(ply_path="a.ply"),
                    material=VBDMaterial(body=VBDSoftBody()),
                ),
                PartConfig(
                    name="also_soft",
                    source=PlySource(ply_path="b.ply"),
                    material=VBDMaterial(body=VBDSoftBody()),
                ),
            ],
            constraints=[PinToBody(
                particles=IndexSelector(part="soft", indices=[0, 1]),
                body="also_soft",
            )],
        )


def test_pin_to_body_accepts_rigid_material_target():
    scene = SceneConfig(
        parts=[
            PartConfig(
                name="soft",
                source=PlySource(ply_path="a.ply"),
                material=VBDMaterial(body=VBDSoftBody()),
            ),
            PartConfig(
                name="pot",
                source=PlySource(ply_path="b.ply"),
                material=VBDMaterial(body=VBDRigidBody(kinematic=True)),
            ),
        ],
        constraints=[PinToBody(
            particles=IndexSelector(part="soft", indices=[0]),
            body="pot",
        )],
    )
    assert len(scene.constraints) == 1


def test_pin_to_body_accepts_rigid_material_type_target():
    scene = SceneConfig(
        parts=[
            PartConfig(
                name="soft",
                source=PlySource(ply_path="a.ply"),
                material=VBDMaterial(body=VBDSoftBody()),
            ),
            PartConfig(
                name="rock",
                source=PlySource(ply_path="b.ply"),
                material=RigidMaterial(),
            ),
        ],
        constraints=[PinToBody(
            particles=IndexSelector(part="soft", indices=[0]),
            body="rock",
        )],
    )
    assert scene.constraints[0].body == "rock"


def test_box_selector_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must be <="):
        BoxSelector(part="p", min=(1.0, 0.0, 0.0), max=(0.0, 1.0, 1.0))


def test_proximity_selector_requires_positive_distance():
    with pytest.raises(ValueError):
        ProximitySelector(part="a", to_surface_of="b", max_distance=0.0)


# ── SimConfig (scene-only after Phase I) ──────────────────────────

def _tiny_scene() -> SceneConfig:
    return SceneConfig(parts=[
        PartConfig(
            name="soft",
            source=PlySource(ply_path="a.ply"),
            material=VBDMaterial(body=VBDSoftBody()),
        ),
    ])


def test_simconfig_requires_scene():
    with pytest.raises(Exception):
        SimConfig(backend=NewtonVBDConfig())


def test_simconfig_scene_ok():
    cfg = SimConfig(backend=NewtonVBDConfig(), scene=_tiny_scene())
    assert cfg.as_scene() is cfg.scene


def test_vbd_rigid_body_kinematic_field_default_false():
    assert VBDRigidBody().kinematic is False
    assert VBDRigidBody(kinematic=True).kinematic is True
