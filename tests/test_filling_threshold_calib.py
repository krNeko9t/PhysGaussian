"""Tests for quantile calibration of particle-filling density thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from physics_sim.preprocessing.filling_threshold_calib import (
    resolve_absolute_thresholds_from_quantiles,
)


def test_resolve_quantiles_matches_numpy_on_occupied_mask():
    grid = np.zeros((2, 2, 2), dtype=np.int32)
    dens = np.zeros((2, 2, 2), dtype=np.float32)
    grid[0, 0, 0] = 1
    grid[1, 1, 1] = 1
    dens[0, 0, 0] = 10.0
    dens[1, 1, 1] = 30.0

    d_abs, s_abs = resolve_absolute_thresholds_from_quantiles(
        grid, dens, density_quantile=0.5, search_quantile=0.25
    )
    samples = np.array([10.0, 30.0], dtype=np.float64)
    assert d_abs == pytest.approx(float(np.quantile(samples, 0.5)))
    assert s_abs == pytest.approx(float(np.quantile(samples, 0.25)))


def test_resolve_falls_back_to_positive_density_when_grid_zero():
    grid = np.zeros((2, 2, 2), dtype=np.int32)
    dens = np.zeros((2, 2, 2), dtype=np.float32)
    dens[0, 1, 0] = 5.0
    dens[1, 0, 1] = 15.0

    d_abs, s_abs = resolve_absolute_thresholds_from_quantiles(
        grid, dens, 0.5, 0.5
    )
    assert d_abs == pytest.approx(10.0)
    assert s_abs == pytest.approx(10.0)


def test_resolve_raises_when_no_samples():
    grid = np.zeros((2, 2, 2), dtype=np.int32)
    dens = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="no occupied"):
        resolve_absolute_thresholds_from_quantiles(grid, dens, 0.5, 0.5)


def test_resolve_shape_mismatch():
    g = np.zeros((2, 2, 2), dtype=np.int32)
    d = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        resolve_absolute_thresholds_from_quantiles(g, d, 0.5, 0.5)


def test_filling_config_quantile_validates_open_interval():
    pytest.importorskip("pydantic")
    from physics_sim.config.models import FillingConfig

    FillingConfig(
        threshold_mode="quantile",
        density_threshold=0.8,
        search_threshold=0.3,
    )
    with pytest.raises(ValueError, match="density_threshold"):
        FillingConfig(
            threshold_mode="quantile",
            density_threshold=1.0,
            search_threshold=0.3,
        )
    with pytest.raises(ValueError, match="search_threshold"):
        FillingConfig(
            threshold_mode="quantile",
            density_threshold=0.5,
            search_threshold=0.0,
        )
