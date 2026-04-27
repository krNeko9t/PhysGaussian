"""Runtime smoke tests for MPM apply_constraints.

Requires GPU / warp / newton; skipped otherwise.  MPM has a narrower
surface in v1 — only PinToWorld is supported (CollideOnly / PinToBody
require the Phase F2 mesh-collider work).
"""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

from physics_sim.backend.newton_mpm.solver import NewtonMPMBackend  # noqa: E402
from physics_sim.backend.spec import MaterialSetupSpec, PartRuntimeInfo  # noqa: E402
from physics_sim.config.models import NewtonMPMConfig, MPMMaterial  # noqa: E402
from physics_sim.scene.constraint_resolver import (  # noqa: E402
    ResolvedCollideOnly, ResolvedPinToBody, ResolvedPinToWorld,
)


def _make_mpm_backend(n_side: int = 4) -> tuple[NewtonMPMBackend, np.ndarray]:
    device = "cuda:0"
    be = NewtonMPMBackend(cfg=NewtonMPMConfig(), device=device)
    xs = np.linspace(0.0, 1.0, n_side)
    pts = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
    positions = torch.from_numpy(pts.astype(np.float32)).to(device)
    volumes = torch.full((positions.shape[0],), 1e-3, device=device)
    covariances = torch.zeros((positions.shape[0], 6), device=device)
    covariances[:, [0, 3, 5]] = 1e-4

    be.initialize(positions, volumes, covariances)
    be.set_material(MaterialSetupSpec(
        gravity=(0.0, -9.8, 0.0),
        per_part=[PartRuntimeInfo(
            name="blob",
            particle_indices=list(range(positions.shape[0])),
            material=MPMMaterial.jelly(),
        )],
    ))
    be.set_boundary_conditions([], None)
    be.finalize()
    return be, pts


def test_mpm_pin_to_world_zeros_mass():
    be, pts = _make_mpm_backend(n_side=4)
    gs_indices = np.nonzero(pts[:, 1] == pts[:, 1].min())[0]
    be.apply_constraints([ResolvedPinToWorld(
        particle_indices=gs_indices.astype(np.int64),
        part_name="blob",
    )])
    rt = be._runtime
    mass = rt.model.particle_mass.numpy()
    inv_mass = rt.model.particle_inv_mass.numpy()
    flags = rt.model.particle_flags.numpy()
    density = rt.solver._mpm_model.particle_density.numpy()
    assert np.allclose(mass[gs_indices], 0.0)
    assert np.allclose(inv_mass[gs_indices], 0.0)
    assert np.allclose(density[gs_indices], 0.0)
    assert np.all(flags[gs_indices] & int(newton.ParticleFlags.ACTIVE))


def test_mpm_pin_to_world_freezes_positions():
    be, pts = _make_mpm_backend(n_side=4)
    gs_indices = np.nonzero(pts[:, 1] == pts[:, 1].min())[0]
    be.apply_constraints([ResolvedPinToWorld(
        particle_indices=gs_indices.astype(np.int64),
        part_name="blob",
    )])
    rt = be._runtime
    pre = rt.state_0.particle_q.numpy()[gs_indices].copy()
    for f in range(10):
        be.pre_step(1e-3, f)
        be.step(1e-3, f)
    post = rt.state_0.particle_q.numpy()[gs_indices]
    np.testing.assert_allclose(post, pre, atol=1e-5)


def test_mpm_collide_only_not_implemented():
    be, _ = _make_mpm_backend(n_side=3)
    with pytest.raises(NotImplementedError, match="CollideOnly"):
        be.apply_constraints([ResolvedCollideOnly(part_name="blob")])


def test_mpm_pin_to_body_not_implemented():
    be, _ = _make_mpm_backend(n_side=3)
    with pytest.raises(NotImplementedError, match="PinToBody"):
        be.apply_constraints([ResolvedPinToBody(
            particle_indices=np.array([0, 1], dtype=np.int64),
            particle_world_positions=np.zeros((2, 3), dtype=np.float64),
            part_name="blob",
            body_part_name="nonexistent",
        )])
