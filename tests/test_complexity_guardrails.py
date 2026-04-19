from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_complexity_guardrail_module_exists_with_expected_thresholds():
    src = _read("physics_sim/complexity_guardrails.py")
    assert "FILE_LINE_LIMIT = 300" in src
    assert "FUNCTION_LINE_LIMIT = 80" in src
    assert "NESTING_DEPTH_LIMIT = 4" in src
    assert "--strict" in src


def test_stage_loop_uses_explicit_diagnostics_contract():
    sim_loop = _read("physics_sim/stages/sim_loop.py")
    backend_base = _read("physics_sim/backend/base.py")
    assert "backend.get_diagnostics()" in sim_loop
    assert "hasattr(backend, \"get_diagnostics\")" not in sim_loop
    assert "def get_diagnostics(self) -> Optional[dict]:" in backend_base


def test_boundary_normalization_is_single_stage_entry():
    backend_init = _read("physics_sim/stages/backend_init.py")
    boundary_norm = _read("physics_sim/stages/boundary_normalization.py")
    assert "normalize_boundary_conditions(" in backend_init
    assert "def normalize_boundary_conditions(" in boundary_norm
