"""Chunked execution helpers for particle filling pipeline."""

from __future__ import annotations

from collections.abc import Callable

from tqdm import tqdm


def run_densify_stage(
    *,
    n_particles: int,
    particle_chunk_size: int,
    progress: bool,
    sync_each_chunk: bool,
    ti_sync: Callable[[], None],
    densify_range_fn: Callable[..., None],
    densify_all_fn: Callable[..., None],
    ti_pos,
    ti_opacity,
    ti_cov,
    grid,
    grid_density,
    grid_dx: float,
    densify_r_cap: int,
) -> None:
    if progress and n_particles > particle_chunk_size:
        for start in tqdm(
            range(0, n_particles, particle_chunk_size),
            desc="particle_filling/densify",
            unit="chunk",
            dynamic_ncols=True,
        ):
            end = min(start + particle_chunk_size, n_particles)
            densify_range_fn(
                ti_pos,
                ti_opacity,
                ti_cov,
                grid,
                grid_density,
                grid_dx,
                start,
                end,
                densify_r_cap,
            )
            if sync_each_chunk:
                ti_sync()
        return

    densify_all_fn(
        ti_pos,
        ti_opacity,
        ti_cov,
        grid,
        grid_density,
        grid_dx,
        densify_r_cap,
    )
    if sync_each_chunk:
        ti_sync()


def run_dense_fill_stage(
    *,
    grid_total: int,
    grid_chunk_size: int,
    progress: bool,
    sync_each_chunk: bool,
    ti_sync: Callable[[], None],
    fill_range_fn: Callable[..., int],
    fill_all_fn: Callable[..., int],
    grid,
    grid_density,
    grid_dx: float,
    density_thres: float,
    particles,
    max_particles_per_cell: int,
    max_samples: int,
) -> int:
    if progress:
        fill_num = 0
        for start in tqdm(
            range(0, grid_total, grid_chunk_size),
            desc="particle_filling/dense",
            unit="chunk",
            dynamic_ncols=True,
        ):
            end = min(start + grid_chunk_size, grid_total)
            fill_num = fill_range_fn(
                grid,
                grid_density,
                grid_dx,
                density_thres,
                particles,
                fill_num,
                max_particles_per_cell,
                start=start,
                end=end,
            )
            if sync_each_chunk:
                ti_sync()
            if fill_num >= max_samples:
                return max_samples
        return fill_num

    fill_num = fill_all_fn(
        grid,
        grid_density,
        grid_dx,
        density_thres,
        particles,
        0,
        max_particles_per_cell,
    )
    if sync_each_chunk:
        ti_sync()
    return min(fill_num, max_samples)


def run_internal_fill_stage(
    *,
    grid_total: int,
    grid_chunk_size: int,
    progress: bool,
    sync_each_chunk: bool,
    ti_sync: Callable[[], None],
    fill_range_fn: Callable[..., int],
    fill_all_fn: Callable[..., int],
    grid,
    grid_density,
    grid_dx: float,
    particles,
    fill_num: int,
    max_particles_per_cell: int,
    search_exclude_dir: int,
    ray_cast_dir: int,
    search_thres: float,
    max_samples: int,
) -> int:
    if progress:
        for start in tqdm(
            range(0, grid_total, grid_chunk_size),
            desc="particle_filling/internal",
            unit="chunk",
            dynamic_ncols=True,
        ):
            end = min(start + grid_chunk_size, grid_total)
            fill_num = fill_range_fn(
                grid,
                grid_density,
                grid_dx,
                particles,
                fill_num,
                max_particles_per_cell,
                exclude_dir=search_exclude_dir,
                ray_cast_dir=ray_cast_dir,
                threshold=search_thres,
                start=start,
                end=end,
            )
            if sync_each_chunk:
                ti_sync()
            if fill_num >= max_samples:
                return max_samples
        return fill_num

    fill_num = fill_all_fn(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        exclude_dir=search_exclude_dir,
        ray_cast_dir=ray_cast_dir,
        threshold=search_thres,
    )
    if sync_each_chunk:
        ti_sync()
    return min(fill_num, max_samples)

