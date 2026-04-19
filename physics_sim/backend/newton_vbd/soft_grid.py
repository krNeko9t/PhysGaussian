"""Soft-grid generation and particle embedding for NewtonVBD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_sim.backend.newton_vbd.barycentric import compute_barycentric


@dataclass
class SoftGridSpec:
    """Regular tet-grid parameters for one soft object."""

    bbox_min: np.ndarray
    dim_x: int
    dim_y: int
    dim_z: int
    cell_x: float
    cell_y: float
    cell_z: float
    density: float
    k_mu: float
    k_lambda: float
    k_damp: float


@dataclass
class SoftGridEmbedding:
    """Precomputed soft grid and barycentric embedding outputs."""

    spec: SoftGridSpec
    grid_vertices: np.ndarray
    tet_cells: np.ndarray
    tet_ids: np.ndarray
    bary_coords: np.ndarray
    outside_count: int

    @property
    def vert_count(self) -> int:
        return int(
            (self.spec.dim_x + 1)
            * (self.spec.dim_y + 1)
            * (self.spec.dim_z + 1)
        )


def compute_soft_grid_spec(pos_np: np.ndarray, material: dict) -> SoftGridSpec:
    """Parse and validate grid/material params for add_soft_grid."""
    if pos_np.ndim != 2 or pos_np.shape[1] != 3 or pos_np.shape[0] == 0:
        raise ValueError("particle subset must be a non-empty (N, 3) array")

    bbox_min = pos_np.min(axis=0).astype(np.float64, copy=False)
    bbox_max = pos_np.max(axis=0).astype(np.float64, copy=False)

    padding = float(material.get("grid_padding", 0.05))
    if padding < 0.0:
        raise ValueError("grid_padding must be >= 0")
    bbox_min = bbox_min - padding
    bbox_max = bbox_max + padding
    extent = bbox_max - bbox_min

    cell_size = material.get("cell_size", None)
    if cell_size is not None:
        cell_size = float(cell_size)
        if cell_size <= 0.0:
            raise ValueError("cell_size must be > 0")
    else:
        grid_res = int(material.get("grid_resolution", 8))
        if grid_res <= 0:
            raise ValueError("grid_resolution must be >= 1")
        max_extent = float(extent.max())
        if max_extent <= 0.0:
            raise ValueError("soft-body bbox extent must be positive")
        cell_size = max_extent / float(grid_res)

    extent_safe = np.maximum(extent, cell_size)
    dim_x = max(1, int(np.ceil(extent_safe[0] / cell_size)))
    dim_y = max(1, int(np.ceil(extent_safe[1] / cell_size)))
    dim_z = max(1, int(np.ceil(extent_safe[2] / cell_size)))

    cell_x = float(extent_safe[0] / dim_x)
    cell_y = float(extent_safe[1] / dim_y)
    cell_z = float(extent_safe[2] / dim_z)

    density = float(material.get("density", 1e3))
    if density <= 0.0:
        raise ValueError("density must be > 0")
    k_mu = float(material.get("k_mu", 1e5))
    k_lambda = float(material.get("k_lambda", 1e5))
    k_damp = float(material.get("k_damp", 1e-3))
    if k_mu < 0.0 or k_lambda < 0.0 or k_damp < 0.0:
        raise ValueError("k_mu, k_lambda, k_damp must be >= 0")

    return SoftGridSpec(
        bbox_min=bbox_min,
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_z=cell_z,
        density=density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=k_damp,
    )


def _build_soft_grid_vertices(spec: SoftGridSpec) -> np.ndarray:
    vert_count = (spec.dim_x + 1) * (spec.dim_y + 1) * (spec.dim_z + 1)
    grid_verts = np.zeros((vert_count, 3), dtype=np.float64)

    vi = 0
    for z in range(spec.dim_z + 1):
        for y in range(spec.dim_y + 1):
            for x in range(spec.dim_x + 1):
                grid_verts[vi] = [
                    x * spec.cell_x + spec.bbox_min[0],
                    y * spec.cell_y + spec.bbox_min[1],
                    z * spec.cell_z + spec.bbox_min[2],
                ]
                vi += 1
    return grid_verts


def _build_soft_tet_cells(dim_x: int, dim_y: int, dim_z: int) -> np.ndarray:
    def grid_index(x: int, y: int, z: int) -> int:
        return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

    tet_list: list[list[int]] = []
    for z in range(dim_z):
        for y in range(dim_y):
            for x in range(dim_x):
                v0 = grid_index(x, y, z)
                v1 = grid_index(x + 1, y, z)
                v2 = grid_index(x + 1, y, z + 1)
                v3 = grid_index(x, y, z + 1)
                v4 = grid_index(x, y + 1, z)
                v5 = grid_index(x + 1, y + 1, z)
                v6 = grid_index(x + 1, y + 1, z + 1)
                v7 = grid_index(x, y + 1, z + 1)

                if (x & 1) ^ (y & 1) ^ (z & 1):
                    tet_list.extend(
                        [
                            [v0, v1, v4, v3],
                            [v2, v3, v6, v1],
                            [v5, v4, v1, v6],
                            [v7, v6, v3, v4],
                            [v4, v1, v6, v3],
                        ]
                    )
                else:
                    tet_list.extend(
                        [
                            [v1, v2, v5, v0],
                            [v3, v0, v7, v2],
                            [v4, v7, v0, v5],
                            [v6, v5, v2, v7],
                            [v5, v2, v7, v0],
                        ]
                    )
    return np.array(tet_list, dtype=np.int32)


def build_soft_grid_embedding(pos_np: np.ndarray, material: dict) -> SoftGridEmbedding:
    spec = compute_soft_grid_spec(pos_np=pos_np, material=material)
    grid_verts = _build_soft_grid_vertices(spec)
    tet_cells = _build_soft_tet_cells(spec.dim_x, spec.dim_y, spec.dim_z)
    tet_ids, bary, outside_count = compute_barycentric(pos_np, grid_verts, tet_cells)
    return SoftGridEmbedding(
        spec=spec,
        grid_vertices=grid_verts,
        tet_cells=tet_cells,
        tet_ids=tet_ids,
        bary_coords=bary,
        outside_count=outside_count,
    )
