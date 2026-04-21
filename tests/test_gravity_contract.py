from __future__ import annotations

from pathlib import Path

import pytest

# Uses real torch internally via normalize_internal_gravity.
pytest.importorskip("torch")

from physics_sim.config.models import MPMMaterial, RigidMaterial
from physics_sim.coord import normalize_internal_gravity
from physics_sim.stages.backend_init import _resolve_gravity
from physics_sim.stages.scene_setup import PartRuntimeInfo


REPO_ROOT = Path(__file__).resolve().parents[1]


def _info(name: str, material) -> PartRuntimeInfo:
    return PartRuntimeInfo(
        name=name,
        particle_indices=[0],
        material=material,
    )


def test_resolve_gravity_default_is_internal_negative_y():
    assert _resolve_gravity([]) == pytest.approx([0.0, -9.8, 0.0], abs=1e-6)


def test_resolve_gravity_reads_g_magnitude_from_first_object():
    g = _resolve_gravity([_info("cube", RigidMaterial(g_magnitude=9.81))])
    assert g == pytest.approx([0.0, -9.81, 0.0], abs=1e-6)


def test_resolve_gravity_absolute_value_only():
    # g_magnitude is always treated as a positive magnitude.
    g = _resolve_gravity([_info("cube", MPMMaterial(g_magnitude=-9.8))])
    assert g == pytest.approx([0.0, -9.8, 0.0], abs=1e-6)


def test_solver_contract_rejects_missing_or_scalar_gravity():
    with pytest.raises(ValueError, match="E_GRAVITY_MISSING"):
        normalize_internal_gravity(
            None,
            backend="newton_rigid",
            config_path="material.g",
            allow_scalar=False,
        )

    with pytest.raises(ValueError, match="E_GRAVITY_SHAPE"):
        normalize_internal_gravity(
            9.8,
            backend="newton_rigid",
            config_path="material.g",
            allow_scalar=False,
        )


def test_gravity_contract_enforced_at_stage_boundary():
    """Gravity validation lives in backend_init.py after Phase B.

    Backends receive gravity as a typed tuple on MaterialSetupSpec;
    they no longer parse raw dict values so the per-backend contract
    checks disappeared from solver source.
    """
    src = (REPO_ROOT / "physics_sim/stages/backend_init.py").read_text(encoding="utf-8")
    assert "gravity_contract_error(" in src
    assert "E_GRAVITY_SHAPE" in src
    assert "(0.0, 0.0, -9.8)" not in src
    assert "[0.0, 0.0, -9.8]" not in src

