"""Unit tests for physics_sim.scene.filling.

The strategy layer is pure PyTorch and runs on CPU.  The end-to-end
``apply_particle_filling`` integration test needs taichi + CUDA for the
underlying ``fill_particles`` Taichi kernels and is skipped otherwise.
"""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from physics_sim.scene.filling import (  # noqa: E402
    FillStrategy,
    NearestNeighborStrategy,
    _group_objects,
    _nearest_src_index,
    _pick_group_config,
    apply_particle_filling,
)
from physics_sim.scene.objects import SceneObject  # noqa: E402


def _make_obj(
    name: str,
    n: int,
    pos_offset: float = 0.0,
    fill_group: str | None = None,
    particle_filling=None,
) -> SceneObject:
    """Tiny CPU SceneObject with dummy attribute tensors."""
    pos = torch.arange(n, dtype=torch.float32).reshape(n, 1).repeat(1, 3) + pos_offset
    return SceneObject(
        name=name,
        role="dynamic",
        positions=pos,
        covariances=torch.zeros(n, 6),
        opacities=torch.full((n, 1), 0.5),
        shs=torch.zeros(n, 16, 3),
        quats=torch.zeros(n, 4),
        scales=torch.zeros(n, 3),
        fill_group=fill_group,
        particle_filling=particle_filling,
    )


# ── _nearest_src_index ──────────────────────────────────────────────────

def test_nearest_src_index_basic():
    src = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    new = torch.tensor([[0.1, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 11.0, 0.0]])
    idx = _nearest_src_index(src, new)
    assert idx.tolist() == [0, 1, 2]


def test_nearest_src_index_chunked_matches_unchunked():
    torch.manual_seed(0)
    src = torch.randn(20, 3)
    new = torch.randn(97, 3)  # non-multiple of any typical chunk
    small = _nearest_src_index(src, new, chunk=8)
    big = _nearest_src_index(src, new, chunk=1024)
    assert torch.equal(small, big)


# ── NearestNeighborStrategy ────────────────────────────────────────────

def test_nearest_neighbor_strategy_gathers_all_attrs():
    src_pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    src_attrs = {
        "shs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "part_id": torch.tensor([0, 1], dtype=torch.long),
    }
    new_pos = torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    out = NearestNeighborStrategy().inherit(
        src_pos=src_pos, src_attrs=src_attrs, new_pos=new_pos,
    )
    assert torch.equal(out["shs"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert out["part_id"].tolist() == [0, 1]


def test_nearest_neighbor_strategy_implements_protocol():
    # Ensure duck-typing satisfies the FillStrategy protocol.
    s: FillStrategy = NearestNeighborStrategy()
    assert callable(s.inherit)


# ── _group_objects ──────────────────────────────────────────────────────

def test_group_objects_none_groups_are_singletons():
    objs = [
        _make_obj("a", 2),
        _make_obj("b", 2),
        _make_obj("c", 2),
    ]
    assert _group_objects(objs) == [[0], [1], [2]]


def test_group_objects_merges_by_name_preserving_first_appearance():
    objs = [
        _make_obj("a", 2, fill_group="plant"),
        _make_obj("b", 2),
        _make_obj("c", 2, fill_group="plant"),
        _make_obj("d", 2, fill_group="other"),
    ]
    groups = _group_objects(objs)
    assert groups == [[0, 2], [1], [3]]


# ── _pick_group_config ─────────────────────────────────────────────────

def test_pick_group_config_none_when_nobody_filled():
    objs = [_make_obj("a", 2), _make_obj("b", 2)]
    assert _pick_group_config(objs, 0) is None


def test_pick_group_config_raises_when_configs_differ():
    pytest.importorskip("pydantic")
    from physics_sim.config.models import FillingConfig
    from physics_sim.errors import PhysicsSimConfigurationError

    objs = [
        _make_obj("a", 2, particle_filling=FillingConfig(n_grid=64)),
        _make_obj("b", 2, particle_filling=FillingConfig(n_grid=128)),
    ]
    with pytest.raises(PhysicsSimConfigurationError, match="fill_group"):
        _pick_group_config(objs, 0)


def test_pick_group_config_returns_shared_cfg():
    pytest.importorskip("pydantic")
    from physics_sim.config.models import FillingConfig

    cfg = FillingConfig(n_grid=96)
    objs = [
        _make_obj("a", 2, particle_filling=cfg),
        _make_obj("b", 2, particle_filling=cfg),
    ]
    assert _pick_group_config(objs, 0) is cfg


# ── apply_particle_filling fast path ────────────────────────────────────

def test_apply_particle_filling_noop_when_nothing_declared():
    objs = [_make_obj("a", 3), _make_obj("b", 2)]
    out = apply_particle_filling(objs)
    assert out == objs  # identity pass-through


# ── SceneConfig fill_group validator ────────────────────────────────────

def test_scene_config_fill_group_validator_requires_consistent_cfg():
    pytest.importorskip("pydantic")
    from physics_sim.config.models import FillingConfig, PlySource
    from physics_sim.config.scene import PartConfig, SceneConfig

    part_a = PartConfig(
        name="a",
        source=PlySource(path="/tmp/a.ply"),
        fill_group="plant",
        particle_filling=FillingConfig(n_grid=64),
    )
    part_b = PartConfig(
        name="b",
        source=PlySource(path="/tmp/b.ply"),
        fill_group="plant",
        particle_filling=FillingConfig(n_grid=128),
    )
    with pytest.raises(ValueError, match="fill_group 'plant'"):
        SceneConfig(parts=[part_a, part_b])


def test_scene_config_fill_group_validator_accepts_matching_cfgs():
    pytest.importorskip("pydantic")
    from physics_sim.config.models import FillingConfig, PlySource
    from physics_sim.config.scene import PartConfig, SceneConfig

    shared = FillingConfig(n_grid=96)
    parts = [
        PartConfig(
            name="a",
            source=PlySource(path="/tmp/a.ply"),
            fill_group="plant",
            particle_filling=shared,
        ),
        PartConfig(
            name="b",
            source=PlySource(path="/tmp/b.ply"),
            fill_group="plant",
            particle_filling=shared,
        ),
    ]
    SceneConfig(parts=parts)  # no raise


# ── Integration smoke (GPU + taichi) ────────────────────────────────────

def test_apply_particle_filling_end_to_end():
    pytest.importorskip("taichi")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for fill_particles")

    from physics_sim.config.models import FillingConfig

    # Solid cube shell: ~1k surface samples near a unit box edge.
    import math  # noqa: F401 — kept in case we extend sampling below

    n_side = 8
    pts = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if i in (0, n_side - 1) or j in (0, n_side - 1) or k in (0, n_side - 1):
                    pts.append((i / (n_side - 1), j / (n_side - 1), k / (n_side - 1)))
    pos = torch.tensor(pts, dtype=torch.float32, device="cuda:0")
    n = pos.shape[0]

    obj = SceneObject(
        name="cube",
        role="dynamic",
        positions=pos,
        covariances=torch.zeros(n, 6, device="cuda:0").index_fill_(1, torch.tensor([0, 3, 5], device="cuda:0"), 1e-3),
        opacities=torch.full((n, 1), 0.5, device="cuda:0"),
        shs=torch.zeros(n, 16, 3, device="cuda:0"),
        quats=torch.zeros(n, 4, device="cuda:0"),
        scales=torch.zeros(n, 3, device="cuda:0"),
        particle_filling=FillingConfig(
            n_grid=32, max_particles_num=50_000, max_particles_per_cell=1,
        ),
    )
    out = apply_particle_filling([obj])
    assert out[0].n_particles >= n  # may fill, may not depending on density; must not shrink
