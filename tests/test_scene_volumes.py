"""Unit tests for physics_sim.scene.volumes.

Ellipsoid and sanity_check_volumes run pure-torch on CPU.  The
occupancy estimator needs taichi (skipped otherwise).
"""

from __future__ import annotations

import logging
import math

import pytest


torch = pytest.importorskip("torch")

from physics_sim.scene.volumes import (  # noqa: E402
    estimate_volume_ellipsoid,
    estimate_volume_occupancy,
    sanity_check_volumes,
)


# ── estimate_volume_ellipsoid ──────────────────────────────────────────

def test_ellipsoid_3dgs_matches_formula():
    scales = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.5, 0.25, 0.125],
        [2.0, 3.0, 4.0],
    ])
    got = estimate_volume_ellipsoid(scales)
    want = torch.tensor([
        4.0 / 3.0 * math.pi * 1.0,
        4.0 / 3.0 * math.pi * 0.5 * 0.25 * 0.125,
        4.0 / 3.0 * math.pi * 2.0 * 3.0 * 4.0,
    ])
    assert torch.allclose(got, want, rtol=1e-6)


def test_ellipsoid_2dgs_uses_min_axis_as_thickness():
    scales = torch.tensor([[0.5, 2.0], [1.0, 1.0]])
    got = estimate_volume_ellipsoid(scales)
    want = torch.tensor([
        4.0 / 3.0 * math.pi * 0.5 * 2.0 * 0.5,  # min axis = 0.5
        4.0 / 3.0 * math.pi * 1.0 * 1.0 * 1.0,
    ])
    assert torch.allclose(got, want, rtol=1e-6)


def test_ellipsoid_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        estimate_volume_ellipsoid(torch.zeros(5))


def test_ellipsoid_rejects_bad_last_dim():
    with pytest.raises(ValueError, match="last dim"):
        estimate_volume_ellipsoid(torch.zeros(5, 4))


def test_ellipsoid_handles_negative_scales_as_abs():
    # Some upstream code may produce log-scales or flipped axes; we stay
    # positive by construction.
    scales = torch.tensor([[-1.0, 2.0, -3.0]])
    got = estimate_volume_ellipsoid(scales)
    assert float(got.item()) == pytest.approx(4.0 / 3.0 * math.pi * 6.0, rel=1e-6)


def test_ellipsoid_preserves_device_dtype():
    scales = torch.ones(4, 3, dtype=torch.float64)
    got = estimate_volume_ellipsoid(scales)
    assert got.dtype == torch.float64
    assert got.shape == (4,)


# ── estimate_volume_occupancy (needs taichi) ───────────────────────────

def test_occupancy_uniform_grid_gives_equal_volumes():
    pytest.importorskip("taichi")
    if not torch.cuda.is_available():
        pytest.skip("occupancy path routes through taichi + torch; needs CUDA")
    # 5x5x5 lattice in a unit cube → one particle per cell at n_grid=5.
    coords = torch.linspace(0.0, 1.0, 5)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    pos = pos.to("cuda:0")
    vol = estimate_volume_occupancy(pos, n_grid=5)
    assert vol.shape == (pos.shape[0],)
    # All cells have exactly one particle → all volumes equal ≈ dx³.
    assert torch.allclose(vol, vol[0].expand_as(vol), rtol=1e-4)
    assert float(vol[0].item()) > 0.0


# ── sanity_check_volumes ───────────────────────────────────────────────

def test_sanity_check_warns_when_ratio_off(caplog):
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # bbox = 1
    vol = torch.tensor([100.0, 100.0])  # total 200, ratio 200x
    with caplog.at_level(logging.WARNING, logger="physics_sim.scene.volumes"):
        sanity_check_volumes(vol, pos, name="test")
    assert any("ratio" in rec.message for rec in caplog.records)


def test_sanity_check_silent_when_ratio_ok(caplog):
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # bbox = 1
    vol = torch.tensor([0.5, 0.5])  # total 1 → ratio 1.0
    with caplog.at_level(logging.WARNING, logger="physics_sim.scene.volumes"):
        sanity_check_volumes(vol, pos, name="test")
    assert not caplog.records


def test_sanity_check_empty_noop():
    sanity_check_volumes(torch.zeros(0), torch.zeros(0, 3), name="empty")  # no raise


def test_sanity_check_degenerate_bbox_warns(caplog):
    # All particles coincident → bbox volume 0.
    pos = torch.zeros(3, 3)
    vol = torch.tensor([1.0, 1.0, 1.0])
    with caplog.at_level(logging.WARNING, logger="physics_sim.scene.volumes"):
        sanity_check_volumes(vol, pos, name="degen")
    assert any("degenerate" in rec.message for rec in caplog.records)
