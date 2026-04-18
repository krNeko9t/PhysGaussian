"""Barycentric embedding helpers for Newton VBD soft bodies."""

from __future__ import annotations

import numpy as np

from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Numerical tolerances for barycentric computation.
_BARY_DET_SINGULAR = 1e-12
_BARY_INV_ABS_MAX = 1e6
_BARY_INSIDE_EPS = 1e-4


def compute_barycentric(
    points: np.ndarray,
    tet_verts: np.ndarray,
    tet_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Find containing tet and barycentric coordinates for each point."""
    P = points.shape[0]
    T = tet_cells.shape[0]

    v0 = tet_verts[tet_cells[:, 0]]
    v1 = tet_verts[tet_cells[:, 1]]
    v2 = tet_verts[tet_cells[:, 2]]
    v3 = tet_verts[tet_cells[:, 3]]

    mat = np.stack([v0 - v3, v1 - v3, v2 - v3], axis=-1)
    inv_mat = np.zeros_like(mat)
    tet_is_degenerate = np.zeros(T, dtype=bool)
    degenerate_count = 0
    for t in range(T):
        det_val = np.linalg.det(mat[t])
        if abs(det_val) < _BARY_DET_SINGULAR:
            inv_mat[t] = np.eye(3)
            tet_is_degenerate[t] = True
            degenerate_count += 1
            continue
        try:
            inv_t = np.linalg.inv(mat[t])
        except np.linalg.LinAlgError:
            inv_t = np.eye(3)
            tet_is_degenerate[t] = True
            degenerate_count += 1
            continue
        if np.abs(inv_t).max() > _BARY_INV_ABS_MAX:
            inv_mat[t] = np.eye(3)
            tet_is_degenerate[t] = True
            degenerate_count += 1
        else:
            inv_mat[t] = inv_t
    if degenerate_count > 0:
        LOGGER.warning(
            "[Barycentric] degenerate_tets=%s/%s, skip unreliable candidates",
            degenerate_count,
            T,
        )

    centroids = (v0 + v1 + v2 + v3) / 4.0
    tet_ids = np.zeros(P, dtype=np.int32)
    bary = np.zeros((P, 4), dtype=np.float32)
    outside_count = 0

    n_cand = min(128, T)
    batch = 4096
    for start in range(0, P, batch):
        end = min(start + batch, P)
        pts = points[start:end]
        B = pts.shape[0]

        dists = np.linalg.norm(centroids[None, :, :] - pts[:, None, :], axis=2)
        if n_cand < T:
            cand_idx = np.argpartition(dists, n_cand, axis=1)[:, :n_cand]
        else:
            cand_idx = np.tile(np.arange(T), (B, 1))

        for i in range(B):
            found = False
            best_tet = -1
            best_min_bary = -np.inf

            for c in range(cand_idx.shape[1]):
                t = cand_idx[i, c]
                if tet_is_degenerate[t]:
                    continue
                p_local = pts[i] - v3[t]
                lam = inv_mat[t] @ p_local
                lam3 = 1.0 - lam[0] - lam[1] - lam[2]
                min_lam = min(lam[0], lam[1], lam[2], lam3)
                if min_lam >= -_BARY_INSIDE_EPS:
                    tet_ids[start + i] = t
                    bary[start + i] = [lam[0], lam[1], lam[2], lam3]
                    found = True
                    break
                if min_lam > best_min_bary:
                    best_min_bary = min_lam
                    best_tet = t

            if not found:
                outside_count += 1
                t = best_tet
                if t < 0 or tet_is_degenerate[t]:
                    non_degen = np.where(~tet_is_degenerate)[0]
                    if len(non_degen) > 0:
                        cd = np.linalg.norm(centroids[non_degen] - pts[i], axis=1)
                        t = non_degen[np.argmin(cd)]
                    else:
                        t = 0
                p_local = pts[i] - v3[t]
                lam = inv_mat[t] @ p_local
                lam3 = 1.0 - lam[0] - lam[1] - lam[2]
                raw = np.array([lam[0], lam[1], lam[2], lam3], dtype=np.float32)
                raw = np.maximum(raw, 0.0)
                s = raw.sum()
                if s > 0:
                    raw /= s
                else:
                    raw[:] = 0.25
                tet_ids[start + i] = t
                bary[start + i] = raw

    return tet_ids, bary, outside_count
