from __future__ import annotations

import numpy as np
import pytest
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_sim.render.sh_colorizer import ShColorizer, compute_view_directions
from physics_sim.render.sh_eval import C0, C1, eval_sh
from physics_sim.sh_contract import (
    assemble_sh_from_ply_features,
    expected_f_rest_feature_count,
    flatten_sh_coeffs,
    restore_sh_coeffs,
    sh_coeff_count,
)


class _DummyCamera:
    def __init__(self) -> None:
        self.camera_center = torch.zeros(3, dtype=torch.float32)


def test_f_rest_count_and_layout_contract():
    sh_degree = 1
    coeffs = sh_coeff_count(sh_degree)
    expected_rest = expected_f_rest_feature_count(sh_degree)

    features_dc = np.array([[[10.0], [20.0], [30.0]]], dtype=np.float32)
    features_rest = np.arange(expected_rest, dtype=np.float32).reshape(1, expected_rest)
    shs = assemble_sh_from_ply_features(features_dc, features_rest, sh_degree)

    assert shs.shape == (1, coeffs, 3)
    np.testing.assert_allclose(shs[0, 0], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(shs[0, 1], np.array([0.0, 3.0, 6.0], dtype=np.float32))


def test_eval_sh_degree_terms_have_expected_baseline():
    dirs = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    for deg in range(5):
        coeffs = sh_coeff_count(deg)
        sh = torch.zeros((1, 3, coeffs), dtype=torch.float32)
        sh[..., 0] = 2.0
        out = eval_sh(deg, sh, dirs)
        assert torch.allclose(out, torch.full_like(out, 2.0 * C0))

    deg1 = torch.zeros((1, 3, sh_coeff_count(1)), dtype=torch.float32)
    deg1[..., 2] = 1.0
    out_deg1 = eval_sh(1, deg1, dirs)
    assert torch.allclose(out_deg1, torch.full_like(out_deg1, C1))


def test_view_direction_transform_order_is_explicit():
    pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    camera_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    rot_z_90 = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    alignment_inv = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    out = compute_view_directions(
        position=pos,
        camera_center=camera_center,
        view_rotations=rot_z_90,
        alignment_inv=alignment_inv,
    )
    expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6)


def test_convert_sh_requires_explicit_full_rotation_shape():
    colorizer = ShColorizer(sh_degree=0)
    shs = torch.zeros((2, 1, 3), dtype=torch.float32)
    pos = torch.zeros((2, 3), dtype=torch.float32)
    camera = _DummyCamera()
    with pytest.raises(ValueError, match="view_rotations must have shape"):
        colorizer.convert_sh(
            shs=shs,
            camera=camera,
            position=pos,
            view_rotations=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        )

    rgb = colorizer.convert_sh(shs=shs, camera=camera, position=pos)
    assert rgb.shape == (2, 3)
    assert torch.allclose(rgb, torch.full_like(rgb, 0.5))


def test_sh_flatten_restore_round_trip_for_preprocessing():
    shs = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    flat = flatten_sh_coeffs(shs)
    restored = restore_sh_coeffs(flat)
    assert flat.shape == (2, 12)
    assert restored.shape == (2, 4, 3)
    assert torch.equal(restored, shs)

