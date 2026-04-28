"""
Particle filling utilities for densifying 3DGS scenes before physics simulation.

Extracted from PhysGaussian/particle_filling/filling_chunk.py.
This module uses Taichi kernels to:
  1. Compute a density field from Gaussian kernels.
  2. Fill dense grid cells with particles.
  3. Fill internal (occluded) grid cells via ray-casting.
"""

import torch
import numpy as np
import taichi as ti
import mcubes

from physics_sim.preprocessing.filling_threshold_calib import (
    resolve_absolute_thresholds_from_quantiles,
)
from physics_sim.preprocessing.internal_fill_axis import (
    void_fill_ray_axis_to_dir_index,
    void_probe_skip_axis_to_dir_index,
)
from physics_sim.preprocessing.particle_filling_chunks import (
    run_dense_fill_stage,
    run_densify_stage,
    run_internal_fill_stage,
)
from physics_sim.sh_contract import flatten_sh_coeffs, restore_sh_coeffs

# Hard safety cap for densify neighborhood radius.
DEFAULT_DENSIFY_R_CAP = 8


# ── Taichi kernels ────────────────────────────────────────────────────────

@ti.func
def compute_density(index, pos, opacity, cov, grid_dx):
    gaussian_weight = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                node_pos = (index + ti.Vector([i, j, k])) * grid_dx
                dist = pos - node_pos
                gaussian_weight += ti.exp(-0.5 * dist.dot(cov @ dist))
    return opacity * gaussian_weight / 8.0


@ti.kernel
def densify_grids(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    r_cap: int,
):
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        x, y, z = pos[0], pos[1], pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        if 0 <= i and i < grid.shape[0] and 0 <= j and j < grid.shape[1] and 0 <= k and k < grid.shape[2]:
            ti.atomic_add(grid[i, j, k], 1)
        else:
            continue
        cov = ti.Matrix([
            [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
            [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
            [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
        ])
        sig, Q = ti.sym_eig(cov)
        sig[0] = ti.max(sig[0], 1e-8)
        sig[1] = ti.max(sig[1], 1e-8)
        sig[2] = ti.max(sig[2], 1e-8)
        sig_mat = ti.Matrix([[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]])
        cov = Q @ sig_mat @ Q.transpose()
        r = 0.0
        for idx in ti.static(range(3)):
            if sig[idx] < 0:
                sig[idx] = ti.sqrt(-sig[idx])
            else:
                sig[idx] = ti.sqrt(sig[idx])
            r = ti.max(r, sig[idx])
        r = ti.ceil(r / grid_dx, dtype=int)
        r = ti.max(0, ti.min(r, r_cap))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (i + dx >= 0 and i + dx < grid_density.shape[0]
                            and j + dy >= 0 and j + dy < grid_density.shape[1]
                            and k + dz >= 0 and k + dz < grid_density.shape[2]):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos, opacity[pi], cov, grid_dx,
                        )
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)


@ti.kernel
def densify_grids_range(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    start: int,
    end: int,
    r_cap: int,
):
    for pi in range(start, end):
        pos = init_particles[pi]
        x, y, z = pos[0], pos[1], pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        if 0 <= i and i < grid.shape[0] and 0 <= j and j < grid.shape[1] and 0 <= k and k < grid.shape[2]:
            ti.atomic_add(grid[i, j, k], 1)
        else:
            continue
        cov = ti.Matrix([
            [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
            [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
            [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
        ])
        sig, Q = ti.sym_eig(cov)
        sig[0] = ti.max(sig[0], 1e-8)
        sig[1] = ti.max(sig[1], 1e-8)
        sig[2] = ti.max(sig[2], 1e-8)
        sig_mat = ti.Matrix([[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]])
        cov = Q @ sig_mat @ Q.transpose()
        r = 0.0
        for idx in ti.static(range(3)):
            if sig[idx] < 0:
                sig[idx] = ti.sqrt(-sig[idx])
            else:
                sig[idx] = ti.sqrt(sig[idx])
            r = ti.max(r, sig[idx])
        r = ti.ceil(r / grid_dx, dtype=int)
        r = ti.max(0, ti.min(r, r_cap))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (i + dx >= 0 and i + dx < grid_density.shape[0]
                            and j + dy >= 0 and j + dy < grid_density.shape[1]
                            and k + dz >= 0 and k + dz < grid_density.shape[2]):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos, opacity[pi], cov, grid_dx,
                        )
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)


@ti.kernel
def fill_dense_grids(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    density_thres: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid_density:
        if grid_density[i, j, k] > density_thres:
            if grid[i, j, k] < max_particles_per_cell:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                for index in range(tmp_start_idx, tmp_start_idx + diff):
                    if index < new_particles.shape[0]:
                        di, dj, dk = ti.random(), ti.random(), ti.random()
                        new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx
    return new_start_idx


@ti.kernel
def fill_dense_grids_range(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    density_thres: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    start: int,
    end: int,
) -> int:
    new_start_idx = start_idx
    n = grid_density.shape[0]
    nn = n * n
    for t in range(start, end):
        i = t // nn
        rem = t - i * nn
        j = rem // n
        k = rem - j * n
        if grid_density[i, j, k] > density_thres:
            if grid[i, j, k] < max_particles_per_cell:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                for index in range(tmp_start_idx, tmp_start_idx + diff):
                    if index < new_particles.shape[0]:
                        di, dj, dk = ti.random(), ti.random(), ti.random()
                        new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx
    return new_start_idx


@ti.func
def collision_search(grid, grid_density, index, dir_type, size, threshold) -> bool:
    # dir_type: 0..5 => +X,-X,+Y,-Y,+Z,-Z (internal grid axes)
    dir = ti.Vector([0, 0, 0])
    if dir_type == 0:   dir[0] = 1
    elif dir_type == 1: dir[0] = -1
    elif dir_type == 2: dir[1] = 1
    elif dir_type == 3: dir[1] = -1
    elif dir_type == 4: dir[2] = 1
    elif dir_type == 5: dir[2] = -1

    flag = False
    index += dir
    i, j, k = index
    while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
        if grid_density[index] > threshold:
            flag = True
            break
        index += dir
        i, j, k = index
    return flag


@ti.func
def collision_times(grid, grid_density, index, dir_type, size, threshold) -> int:
    dir = ti.Vector([0, 0, 0])
    times = 0
    if dir_type > 5 or dir_type < 0:
        times = 1
    else:
        if dir_type == 0:   dir[0] = 1
        elif dir_type == 1: dir[0] = -1
        elif dir_type == 2: dir[1] = 1
        elif dir_type == 3: dir[1] = -1
        elif dir_type == 4: dir[2] = 1
        elif dir_type == 5: dir[2] = -1

        state = grid[index] > 0
        index += dir
        i, j, k = index
        while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
            new_state = grid_density[index] > threshold
            if new_state != state and state == False:
                times += 1
            state = new_state
            index += dir
            i, j, k = index
    return times


@ti.kernel
def internal_filling(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid:
        if grid[i, j, k] == 0:
            collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid, grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type, size=grid.shape[0], threshold=threshold,
                    )
                    collision_hit = collision_hit and hit_test
            if collision_hit:
                hit_times = collision_times(
                    grid=grid, grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir, size=grid.shape[0], threshold=threshold,
                )
                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    for index in range(tmp_start_idx, tmp_start_idx + diff):
                        if index < new_particles.shape[0]:
                            di, dj, dk = ti.random(), ti.random(), ti.random()
                            new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx
    return new_start_idx


@ti.kernel
def internal_filling_range(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
    start: int,
    end: int,
) -> int:
    new_start_idx = start_idx
    n = grid.shape[0]
    nn = n * n
    for t in range(start, end):
        i = t // nn
        rem = t - i * nn
        j = rem // n
        k = rem - j * n
        if grid[i, j, k] == 0:
            collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid, grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type, size=n, threshold=threshold,
                    )
                    collision_hit = collision_hit and hit_test
            if collision_hit:
                hit_times = collision_times(
                    grid=grid, grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir, size=n, threshold=threshold,
                )
                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    for index in range(tmp_start_idx, tmp_start_idx + diff):
                        if index < new_particles.shape[0]:
                            di, dj, dk = ti.random(), ti.random(), ti.random()
                            new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx
    return new_start_idx


@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


@ti.kernel
def compute_particle_volume(
    pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]


# ── Python-level API ──────────────────────────────────────────────────────

def get_particle_volume(pos, grid_n: int, grid_dx: float, uniform: bool = False):
    """Compute per-particle volume based on grid occupancy."""
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if uniform:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()


def fill_particles(
    pos,
    opacity,
    cov,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres=2.0,
    search_thres=1.0,
    threshold_mode: str = "absolute",
    max_particles_per_cell=1,
    void_probe_skip_axis: str | None = None,
    void_fill_ray_axis: str = "+Z",
    boundary: list = None,
    smooth: bool = False,
    progress: bool = True,
    particle_chunk_size: int = 50_000,
    grid_chunk_size: int = 50_000,
    densify_r_cap: int = DEFAULT_DENSIFY_R_CAP,
    sync_each_chunk: bool = True,
):
    """Fill internal voids of a Gaussian point cloud with additional particles.

    ``threshold_mode``:
        ``absolute`` — ``density_thres`` / ``search_thres`` are direct cutoffs in
        ``grid_density`` units.
        ``quantile`` — both arguments are ``q`` in ``(0, 1)``; after densify,
        absolute cutoffs are taken from ``np.quantile`` over occupied voxels
        (``grid > 0``, else ``grid_density > 0``).

    ``void_probe_skip_axis`` / ``void_fill_ray_axis`` use internal signed axes
    (``+Z``, ``-Y``, …) or aliases ``up``/``down``/``left``/``right``; see
    ``internal_fill_axis``.

    Returns:
        torch.Tensor of shape (N_original + N_filled, 3) on CUDA.
    """
    search_exclude_dir = void_probe_skip_axis_to_dir_index(void_probe_skip_axis)
    ray_cast_dir = void_fill_ray_axis_to_dir_index(void_fill_ray_axis)

    pos_clone = pos.clone()
    if boundary is not None:
        if len(boundary) != 6:
            raise ValueError(
                f"boundary must contain 6 scalars [xmin,xmax,ymin,ymax,zmin,zmax], got {boundary}"
            )
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool).cuda()
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])
        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]
        grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]]).cuda()
        pos = pos - new_origin

    from taichi.lang import impl as _ti_impl_fill

    _rt_fill = _ti_impl_fill.get_runtime()
    if _rt_fill is None or getattr(_rt_fill, "prog", None) is None:
        ti.init(arch=ti.cuda, device_memory_GB=8.0)

    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
    particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
    fill_num = 0

    # 1. compute density field
    n_particles = pos.shape[0]
    run_densify_stage(
        n_particles=n_particles,
        particle_chunk_size=particle_chunk_size,
        progress=progress,
        sync_each_chunk=sync_each_chunk,
        ti_sync=ti.sync,
        densify_range_fn=densify_grids_range,
        densify_all_fn=densify_grids,
        ti_pos=ti_pos,
        ti_opacity=ti_opacity,
        ti_cov=ti_cov,
        grid=grid,
        grid_density=grid_density,
        grid_dx=grid_dx,
        densify_r_cap=densify_r_cap,
    )

    if threshold_mode == "quantile":
        density_thres, search_thres = resolve_absolute_thresholds_from_quantiles(
            grid.to_numpy(),
            grid_density.to_numpy(),
            density_thres,
            search_thres,
        )
        if progress:
            print(
                f"[particle_filling] quantile calibration: "
                f"density_thres={density_thres:.6g} search_thres={search_thres:.6g}"
            )
    elif threshold_mode != "absolute":
        raise ValueError(
            f"threshold_mode must be 'absolute' or 'quantile', got {threshold_mode!r}"
        )

    # 2. fill dense grids
    grid_total = grid_n * grid_n * grid_n
    fill_num = run_dense_fill_stage(
        grid_total=grid_total,
        grid_chunk_size=grid_chunk_size,
        progress=progress,
        sync_each_chunk=sync_each_chunk,
        ti_sync=ti.sync,
        fill_range_fn=fill_dense_grids_range,
        fill_all_fn=fill_dense_grids,
        grid=grid,
        grid_density=grid_density,
        grid_dx=grid_dx,
        density_thres=density_thres,
        particles=particles,
        max_particles_per_cell=max_particles_per_cell,
        max_samples=max_samples,
    )
    if fill_num >= max_samples:
        print(f"[particle_filling] WARNING: filled particles ({fill_num}) exceed max_particles_num ({max_samples}); clamping.")
        fill_num = max_samples
    print("after dense grids: ", fill_num)

    # 3. smooth density field (optional)
    if smooth:
        df = grid_density.to_numpy()
        smoothed_df = mcubes.smooth(df, method="constrained", max_iters=500).astype(np.float32)
        grid_density.from_numpy(smoothed_df)
        print("smooth finished")

    # 4. fill internal grids
    fill_num = run_internal_fill_stage(
        grid_total=grid_total,
        grid_chunk_size=grid_chunk_size,
        progress=progress,
        sync_each_chunk=sync_each_chunk,
        ti_sync=ti.sync,
        fill_range_fn=internal_filling_range,
        fill_all_fn=internal_filling,
        grid=grid,
        grid_density=grid_density,
        grid_dx=grid_dx,
        particles=particles,
        fill_num=fill_num,
        max_particles_per_cell=max_particles_per_cell,
        search_exclude_dir=search_exclude_dir,
        ray_cast_dir=ray_cast_dir,
        search_thres=search_thres,
        max_samples=max_samples,
    )
    print("after internal grids: ", fill_num)

    # 5. combine original + filled particles
    particles_tensor = particles.to_torch()[:fill_num].cuda()
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
    particles_tensor = torch.cat([pos_clone, particles_tensor], dim=0)
    return particles_tensor


@ti.kernel
def get_attr_from_closest(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_cov: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_cov: ti.template(),
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_cov[pi] = ti_cov[min_idx]


def init_filled_particles(pos, shs, cov, opacity, new_pos):
    """Initialise SH / opacity / covariance for filled particles from their nearest originals."""
    shs_flat = flatten_sh_coeffs(shs)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_shs = ti.Vector.field(n=shs_flat.shape[1], dtype=float, shape=shs_flat.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_cov.from_torch(cov.reshape(-1, 6))
    ti_shs.from_torch(shs_flat)
    ti_opacity.from_torch(opacity.reshape(-1))

    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs_flat.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_cov = ti.Vector.field(n=6, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))

    get_attr_from_closest(
        ti_pos, ti_shs, ti_opacity, ti_cov,
        ti_new_pos, ti_new_shs, ti_new_opacity, ti_new_cov,
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    cov_tensor = ti_new_cov.to_torch().cuda()

    shs_tensor = torch.cat([shs_flat, shs_tensor], dim=0)
    shs_tensor = restore_sh_coeffs(shs_tensor)
    opacity_tensor = torch.cat([opacity, opacity_tensor.reshape(-1, 1)], dim=0)
    cov_tensor = torch.cat([cov, cov_tensor], dim=0)
    return shs_tensor, opacity_tensor, cov_tensor
