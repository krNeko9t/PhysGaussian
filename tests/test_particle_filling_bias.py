"""GPU unit test for Stage-2 density-biased offset.

Verifies that ``_density_biased_cell_offset`` in
``physics_sim.preprocessing.particle_filling`` leans the sampled offset
toward the denser face-adjacent neighbor cell, and falls back to uniform
when neighbors on both sides are equally dense.

Skipped without taichi + CUDA.

Note: Avoid ``from __future__ import annotations`` here: Taichi reads kernel
parameter annotations at import time; postponed annotations (PEP 563) store
strings, which fail ``Kernel.extract_arguments`` validation.
"""

import pytest


torch = pytest.importorskip("torch")
ti = pytest.importorskip("taichi")


N_SAMPLES = 10_000
GRID_N = 9
CENTER = (4, 4, 4)


def _ensure_taichi_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for taichi kernels")
    from taichi.lang import impl as _ti_impl
    rt = _ti_impl.get_runtime()
    if rt is None or getattr(rt, "prog", None) is None:
        ti.init(arch=ti.cuda, device_memory_GB=1.0)


def _run_bias_kernel(grid_density, floor: float):
    """Compile + run a sampling kernel that calls the unit under test."""
    from physics_sim.preprocessing.particle_filling import (  # noqa: E402
        _density_biased_cell_offset,
    )
    offsets = ti.Vector.field(n=3, dtype=float, shape=N_SAMPLES)
    ci, cj, ck = CENTER

    @ti.kernel
    def _sample(fl: float):
        for idx in range(N_SAMPLES):
            offsets[idx] = _density_biased_cell_offset(
                grid_density, ci, cj, ck, fl,
            )

    _sample(floor)
    return offsets.to_numpy()


def test_asymmetric_density_biases_toward_dense_side() -> None:
    """Material on -x side of center → offsets cluster at x<0.5."""
    _ensure_taichi_cuda()
    grid_density = ti.field(dtype=float, shape=(GRID_N, GRID_N, GRID_N))
    grid_density.fill(0.0)
    # Wall of dense cells on the -x side of the center cell.
    for di in range(1, 4):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                grid_density[
                    CENTER[0] - di, CENTER[1] + dj, CENTER[2] + dk
                ] = 200.0

    offsets = _run_bias_kernel(grid_density, floor=20.0)
    frac_left = float((offsets[:, 0] < 0.5).mean())
    # With d_neg=200, d_pos=0, floor=20 → p_left ≈ 220/240 ≈ 0.917.
    # Allow a wide margin for random variance; the key point is > 0.70.
    assert frac_left > 0.70, (
        f"expected >70% offsets on dense -x side, got {frac_left:.2%}"
    )


def test_symmetric_density_is_near_uniform() -> None:
    """All neighbors equally dense → axis-wise bias ≈ 50/50."""
    _ensure_taichi_cuda()
    grid_density = ti.field(dtype=float, shape=(GRID_N, GRID_N, GRID_N))
    grid_density.fill(100.0)

    offsets = _run_bias_kernel(grid_density, floor=20.0)
    for axis in range(3):
        frac_hi = float((offsets[:, axis] >= 0.5).mean())
        assert 0.45 <= frac_hi <= 0.55, (
            f"axis {axis}: expected ~0.5, got {frac_hi:.2%}"
        )


def test_empty_neighbors_draw_uniform() -> None:
    """All neighbors at 0 density → floor alone drives p=0.5 → uniform."""
    _ensure_taichi_cuda()
    grid_density = ti.field(dtype=float, shape=(GRID_N, GRID_N, GRID_N))
    grid_density.fill(0.0)

    offsets = _run_bias_kernel(grid_density, floor=20.0)
    for axis in range(3):
        frac_hi = float((offsets[:, axis] >= 0.5).mean())
        assert 0.45 <= frac_hi <= 0.55, (
            f"axis {axis}: zero-density fallback should be uniform, got {frac_hi:.2%}"
        )
