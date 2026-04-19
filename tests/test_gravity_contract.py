from __future__ import annotations

import logging
from pathlib import Path

import pytest

from physics_sim.coord import normalize_internal_gravity
from physics_sim.stages.backend_init import _resolve_gravity


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_gravity_default_is_internal_negative_y():
    assert _resolve_gravity([]) == pytest.approx([0.0, -9.8, 0.0], abs=1e-6)


def test_resolve_gravity_scalar_is_normalized_and_logged(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    g = _resolve_gravity([
        {"name": "cube", "material": {"g": 9.81}},
    ])
    assert g == pytest.approx([0.0, -9.81, 0.0], abs=1e-6)

    text = "\n".join(rec.message for rec in caplog.records)
    assert "[Gravity][backend=backend_init]" in text
    assert "material=cube" in text
    assert "config_path=per_object[0].material.g" in text


def test_resolve_gravity_vector_must_follow_internal_y_up_contract():
    with pytest.raises(ValueError, match="E_GRAVITY_AXIS"):
        _resolve_gravity([
            {"name": "cube", "material": {"g": [0.0, 0.0, -9.8]}},
        ])


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
        "physics_sim/backend/newton_rigid/solver.py",
        "physics_sim/backend/newton_vbd/solver.py",
        "physics_sim/backend/newton_mpm/materials.py",
    )
    for rel_path in solver_paths:
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "normalize_internal_gravity(" in src
        assert "E_GRAVITY_MISSING" in src
        assert "(0.0, 0.0, -9.8)" not in src
        assert "[0.0, 0.0, -9.8]" not in src
