from __future__ import annotations

import logging
from pathlib import Path

import pytest

from physics_sim.config.models import MPMMaterial, RigidMaterial
from physics_sim.coord import normalize_internal_gravity
from physics_sim.stages.backend_init import _resolve_gravity
from physics_sim.stages.scene_setup import ObjectRuntimeInfo


REPO_ROOT = Path(__file__).resolve().parents[1]


def _info(name: str, material) -> ObjectRuntimeInfo:
    return ObjectRuntimeInfo(
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


def test_solver_files_enforce_contract_without_z_up_defaults():
    solver_paths = (
        "physics_sim/backend/newton_rigid/materials.py",
        "physics_sim/backend/newton_vbd/solver.py",
        "physics_sim/backend/newton_mpm/materials.py",
    )
    for rel_path in solver_paths:
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "normalize_internal_gravity(" in src
        assert "E_GRAVITY_MISSING" in src
        assert "(0.0, 0.0, -9.8)" not in src
        assert "[0.0, 0.0, -9.8]" not in src

