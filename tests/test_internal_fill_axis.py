"""Unit tests for internal void-fill axis parsing."""

from __future__ import annotations

import pytest

from physics_sim.preprocessing.internal_fill_axis import (
    normalize_fill_axis,
    normalize_fill_axis_optional,
    void_fill_ray_axis_to_dir_index,
    void_probe_skip_axis_to_dir_index,
)


def test_normalize_signed_axes():
    assert normalize_fill_axis("+z") == "+Z"
    assert normalize_fill_axis("-Y") == "-Y"


def test_normalize_aliases():
    assert normalize_fill_axis("up") == "+Y"
    assert normalize_fill_axis("DOWN") == "-Y"
    assert normalize_fill_axis("right") == "+X"
    assert normalize_fill_axis("Left") == "-X"


def test_normalize_unknown_raises():
    with pytest.raises(ValueError, match="Unknown internal fill axis"):
        normalize_fill_axis("forward")


def test_normalize_optional_blank():
    assert normalize_fill_axis_optional(None) is None
    assert normalize_fill_axis_optional("  ") is None
    assert normalize_fill_axis_optional("-Z") == "-Z"


def test_dir_indices_match_taichi_encoding():
    assert void_probe_skip_axis_to_dir_index(None) == -1
    assert void_probe_skip_axis_to_dir_index("-Z") == 5
    assert void_fill_ray_axis_to_dir_index("+Z") == 4


def test_filling_config_validates_axes():
    pytest.importorskip("pydantic")
    from physics_sim.config.models import FillingConfig

    cfg = FillingConfig(void_probe_skip_axis="down", void_fill_ray_axis="up")
    assert cfg.void_probe_skip_axis == "-Y"
    assert cfg.void_fill_ray_axis == "+Y"
