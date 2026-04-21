"""Tests for ``physics_sim.scene.constraint_resolver``.

These tests mock out ``SceneObject`` and ``ObjectRuntimeInfo`` with
tiny duck-typed classes so that no torch / warp / GPU dependencies are
needed.  They verify:

- ProximitySelector picks by distance-to-mesh surface.
- BoxSelector picks by AABB test.
- IndexSelector validates bounds and offsets into the global model.
- CollideOnly passes through.
- Empty selection raises an error.
- Selector whose source part has no global offset raises an error.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _stub_torch() -> None:
    if "torch" in sys.modules:
        return
    mod = types.ModuleType("torch")
    mod.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = mod


_stub_torch()


from physics_sim.config.models import (  # noqa: E402
    PlySource,
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
from physics_sim.scene.constraint_resolver import (  # noqa: E402
    ResolvedCollideOnly,
    ResolvedPinToBody,
    ResolvedPinToWorld,
    resolve_constraints,
)


# ── Minimal fakes (avoid torch dependency) ─────────────────────────

class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeSceneObject:
    def __init__(self, name: str, positions: np.ndarray) -> None:
        self.name = name
        self.positions = _FakeTensor(positions.astype(np.float64))


class _FakeRuntimeInfo:
    def __init__(self, name: str, indices: list[int]) -> None:
        self.name = name
        self.particle_indices = indices


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def pot_and_branches():
    pot_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float64)
    branch_pos = np.array([
        [0.0, 0.05, 0.0], [1.0, 0.05, 0.0], [0.0, 0.05, 1.0], [1.0, 0.05, 1.0],  # near pot top
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],      # far above
    ], dtype=np.float64)
    scene_objects = [
        _FakeSceneObject("pot", pot_pos),
        _FakeSceneObject("branches", branch_pos),
    ]
    # pot = collide-only (no sim particles); branches gets the 8 global slots 0..7
    objects_runtime = [_FakeRuntimeInfo("branches", list(range(0, 8)))]
    return scene_objects, objects_runtime


# ── Selector resolution ────────────────────────────────────────────

def test_proximity_selector_picks_near_particles(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToWorld(particles=ProximitySelector(
            part="branches", to_surface_of="pot", max_distance=0.15,
        ))],
    )
    resolved = resolve_constraints(scene, scene_objects, objs_runtime)
    assert len(resolved) == 1
    r = resolved[0]
    assert isinstance(r, ResolvedPinToWorld)
    # First 4 branch particles are at y=0.05 (~0.05 from pot's y=0)
    assert sorted(r.particle_indices.tolist()) == [0, 1, 2, 3]


def test_box_selector_picks_inside_aabb(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToWorld(particles=BoxSelector(
            part="branches",
            min=(-0.1, -0.1, -0.1), max=(1.1, 0.5, 1.1),  # only the near ring
        ))],
    )
    resolved = resolve_constraints(scene, scene_objects, objs_runtime)
    r = resolved[0]
    assert isinstance(r, ResolvedPinToWorld)
    assert sorted(r.particle_indices.tolist()) == [0, 1, 2, 3]


def test_index_selector_offsets_to_global(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToWorld(particles=IndexSelector(
            part="branches", indices=[5, 7],
        ))],
    )
    resolved = resolve_constraints(scene, scene_objects, objs_runtime)
    r = resolved[0]
    # Global indices = local + 0 (branches starts at global 0)
    assert sorted(r.particle_indices.tolist()) == [5, 7]


def test_index_selector_rejects_out_of_range(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToWorld(particles=IndexSelector(
            part="branches", indices=[100],  # out of [0, 8)
        ))],
    )
    with pytest.raises(ValueError, match="out of range"):
        resolve_constraints(scene, scene_objects, objs_runtime)


def test_empty_selection_raises(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToWorld(particles=ProximitySelector(
            part="branches", to_surface_of="pot", max_distance=1e-6,  # too strict
        ))],
    )
    with pytest.raises(ValueError, match="zero particles"):
        resolve_constraints(scene, scene_objects, objs_runtime)


# ── PinToBody carries world positions and body ref ────────────────

def test_pin_to_body_resolved_carries_world_positions(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToBody(
            particles=IndexSelector(part="branches", indices=[0, 1]),
            body="pot",
        )],
    )
    resolved = resolve_constraints(scene, scene_objects, objs_runtime)
    r = resolved[0]
    assert isinstance(r, ResolvedPinToBody)
    assert r.body_part_name == "pot"
    assert r.particle_world_positions.shape == (2, 3)
    # Expected y=0.05 for both
    assert np.allclose(r.particle_world_positions[:, 1], 0.05)


# ── CollideOnly passthrough ───────────────────────────────────────

def test_collide_only_resolves_to_part_name(pot_and_branches):
    scene_objects, objs_runtime = pot_and_branches
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[CollideOnly(part="pot")],
    )
    resolved = resolve_constraints(scene, scene_objects, objs_runtime)
    assert isinstance(resolved[0], ResolvedCollideOnly)
    assert resolved[0].part_name == "pot"


# ── Selector on non-sim part rejected ─────────────────────────────

def test_selector_on_collider_only_part_rejected(pot_and_branches):
    """A selector whose ``.part`` has no global offset (non-sim part)
    should raise, because we can't produce global particle indices."""
    scene_objects, objs_runtime = pot_and_branches
    # pot has no entry in objs_runtime, so it has no global_offset.
    scene = SceneConfig(
        parts=[
            PartConfig(name="pot", source=PlySource(ply_path="p.ply"),
                       material=VBDMaterial(body=VBDRigidBody(kinematic=True))),
            PartConfig(name="branches", source=PlySource(ply_path="b.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
        ],
        constraints=[PinToWorld(particles=IndexSelector(part="pot", indices=[0]))],
    )
    with pytest.raises(ValueError, match="not a simulated part"):
        resolve_constraints(scene, scene_objects, objs_runtime)


# ── Scale / performance regression ─────────────────────────────────

def test_proximity_selector_scales_to_hundred_thousand_points():
    """Regression guard: O(N*M) brute force on this size would timeout /
    blow memory.  With cKDTree it should finish in well under a second."""
    import time
    rng = np.random.default_rng(0)
    n_src, n_ref = 100_000, 50_000
    src = rng.random((n_src, 3)) * 10.0
    # Reference cloud near origin: any src within 0.5 of origin should match.
    ref = rng.random((n_ref, 3)) * 0.5
    scene_objects = [
        _FakeSceneObject("src", src),
        _FakeSceneObject("ref", ref),
    ]
    objs_runtime = [_FakeRuntimeInfo("src", list(range(n_src)))]
    scene = SceneConfig(
        parts=[
            PartConfig(name="src", source=PlySource(ply_path="a.ply"),
                       material=VBDMaterial(body=VBDSoftBody())),
            PartConfig(name="ref", source=PlySource(ply_path="b.ply")),
        ],
        constraints=[PinToWorld(particles=ProximitySelector(
            part="src", to_surface_of="ref", max_distance=0.1,
        ))],
    )
    t0 = time.perf_counter()
    resolved = resolve_constraints(scene, scene_objects, objs_runtime)
    dt = time.perf_counter() - t0
    # On CI (no GPU, modest CPU) this should be <1s.  Brute force on this
    # size would allocate a ~40 GB matrix and never finish.
    assert dt < 5.0, f"ProximitySelector took {dt:.2f}s — did it regress to brute force?"
    assert resolved[0].particle_indices.size > 0
