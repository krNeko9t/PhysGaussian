from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_SIM_ROOT = REPO_ROOT / "physics_sim"


def test_pipeline_no_longer_imports_gaussian_renderer():
    src = (REPO_ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert "from physics_sim.renderer.gs_renderer import GaussianRenderer" not in src
    assert "create_scene_asset_loader" in src or "PipelineOrchestrator" in src


def test_scene_setup_depends_on_loader_protocol_not_renderer():
    src = (REPO_ROOT / "physics_sim/stages/scene_setup.py").read_text(encoding="utf-8")
    assert "SceneAssetLoader" in src
    assert "GaussianRenderer" not in src


def test_scene_assembler_uses_typed_gaussian_asset():
    src = (REPO_ROOT / "physics_sim/scene/assembler.py").read_text(encoding="utf-8")
    assert "GaussianAsset" in src
    assert ".load_ply(" in src


def test_camera_mode_registry_has_all_builtin_modes():
    src = (REPO_ROOT / "physics_sim/render/registries.py").read_text(encoding="utf-8")
    for mode in ("json", "orbit", "fixed"):
        assert f'register_camera_mode(\n    "{mode}"' in src


def test_legacy_renderer_facade_is_removed():
    assert not (PHYSICS_SIM_ROOT / "renderer/gs_renderer.py").exists()


def test_no_legacy_renderer_imports():
    forbidden = (
        "physics_sim.renderer.gs_renderer",
        "physics_sim.renderer.backend_gsplat",
        "physics_sim.renderer.backend_diffrast",
        "physics_sim.renderer.backend_base",
    )
    for py in PHYSICS_SIM_ROOT.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in src, f"{py} contains forbidden import token: {token}"


def test_legacy_renderer_package_has_no_python_sources():
    renderer_dir = PHYSICS_SIM_ROOT / "renderer"
    if not renderer_dir.exists():
        return
    assert list(renderer_dir.glob("*.py")) == []
