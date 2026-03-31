"""
Newton VBD backend — unified rigid + soft body simulation.

Uses a **single** ``ModelBuilder`` and ``SolverVBD`` to handle both
rigid bodies (``add_body`` + ``add_shape_*``) and FEM soft bodies
(``add_soft_grid``) in one scene, with automatic contact handling.

Each object in the config declares ``"physics": "rigid"`` or
``"physics": "soft"``.  Rigid bodies reuse the same collision-geometry
strategies as ``NewtonRigidBackend``.  Soft bodies use a **regular
tetrahedral grid** (Newton's ``add_soft_grid``) covering the GS
particle bounding box — this produces uniform, well-conditioned
elements that Newton's VBD solver is optimized for.

GS particles are **embedded** inside the tet grid via barycentric
coordinates, so the coarse FEM grid drives the dense GS point cloud.

Lifecycle (called by the pipeline):
    initialize  → store particles, create ModelBuilder
    set_material → create rigid/soft bodies per object
    set_boundary_conditions → add ground planes / walls
    finalize    → builder.color(), finalize, create VBD solver
    step        → clear forces, collide, solver step, swap states
    get_state   → rigid: broadcast R,t;  soft: bary-interp from tets
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverVBD

from physics_sim.backend.base import PhysicsBackend, SimulationState

# Geometry helpers
from physics_sim.geometry.convex_hull import compute_convex_hull
from physics_sim.geometry.primitives import fit_obb, fit_ellipsoid

try:
    from physics_sim.geometry.alpha_shape import compute_alpha_shape
    _HAS_ALPHA_SHAPE = True
except ImportError:
    _HAS_ALPHA_SHAPE = False


# ── Shared helpers (same as newton_rigid) ────────────────────────────

def _rotmat_to_wp_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → Warp quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (x, y, z, w) → 3×3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = torch.zeros((3, 3), device=q.device, dtype=q.dtype)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _compose_body_quat_wxyz(
    body_quat_xyzw: torch.Tensor,
    init_quats_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Compose a single body quaternion with per-particle initial quaternions.

    ``body_quat_xyzw`` is (4,) in Newton's xyzw convention.
    ``init_quats_wxyz`` is (N, 4) in 3DGS wxyz convention.
    Returns (N, 4) in wxyz: ``q_body * q_init``.
    """
    bx, by, bz, bw = body_quat_xyzw[0], body_quat_xyzw[1], body_quat_xyzw[2], body_quat_xyzw[3]
    iw, ix, iy, iz = init_quats_wxyz[:, 0], init_quats_wxyz[:, 1], init_quats_wxyz[:, 2], init_quats_wxyz[:, 3]
    ow = bw * iw - bx * ix - by * iy - bz * iz
    ox = bw * ix + bx * iw + by * iz - bz * iy
    oy = bw * iy - bx * iz + by * iw + bz * ix
    oz = bw * iz + bx * iy - by * ix + bz * iw
    out = torch.stack([ow, ox, oy, oz], dim=1)
    return torch.nn.functional.normalize(out, dim=1)


def _unpack_cov6_to_3x3(cov6: torch.Tensor) -> torch.Tensor:
    """(N, 6) upper-triangle → (N, 3, 3) symmetric matrix."""
    N = cov6.shape[0]
    out = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    out[:, 0, 0] = cov6[:, 0]; out[:, 0, 1] = cov6[:, 1]; out[:, 0, 2] = cov6[:, 2]
    out[:, 1, 0] = cov6[:, 1]; out[:, 1, 1] = cov6[:, 3]; out[:, 1, 2] = cov6[:, 4]
    out[:, 2, 0] = cov6[:, 2]; out[:, 2, 1] = cov6[:, 4]; out[:, 2, 2] = cov6[:, 5]
    return out


def _pack_cov3x3_to_6(cov3x3: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) symmetric matrix → (N, 6) upper-triangle."""
    out = torch.zeros((cov3x3.shape[0], 6), device=cov3x3.device, dtype=cov3x3.dtype)
    out[:, 0] = cov3x3[:, 0, 0]; out[:, 1] = cov3x3[:, 0, 1]; out[:, 2] = cov3x3[:, 0, 2]
    out[:, 3] = cov3x3[:, 1, 1]; out[:, 4] = cov3x3[:, 1, 2]; out[:, 5] = cov3x3[:, 2, 2]
    return out


# ── Per-object data structures ───────────────────────────────────────

@dataclass
class _RigidInfo:
    """Back-mapping data for a rigid body."""
    body_idx: int
    particle_indices: list[int]
    init_local_pos: torch.Tensor    # (N, 3)  body-local coords
    init_cov_3x3: torch.Tensor      # (N, 3, 3)
    init_quats: Optional[torch.Tensor] = None   # (N, 4) wxyz — for 2DGS
    init_scales: Optional[torch.Tensor] = None   # (N, 2|3) — for 2DGS


@dataclass
class _SoftInfo:
    """Back-mapping data for an FEM soft body."""
    particle_indices: list[int]
    tet_ids: np.ndarray              # (N,)  which tet each GS particle is in
    bary_coords: np.ndarray          # (N, 4) barycentric weights
    vert_offset: int                 # offset in global particle_q
    vert_count: int                  # number of tet mesh vertices
    tet_cells: np.ndarray            # (T, 4)  for computing deformation gradient
    rest_verts: np.ndarray           # (V, 3)  rest-pose tet vertices
    init_cov_6: torch.Tensor         # (N, 6)


# ── Embedding: GS particles inside tet mesh ──────────────────────────

# Numerical tolerances for barycentric computation.
# These are machine-precision-derived values, NOT physics parameters.
# They should NOT need per-scene tuning.

# Determinant below this → tet is truly singular (flat plane).
_BARY_DET_SINGULAR = 1e-12
# Max absolute value of inv(shape_matrix) entries.  Above this the
# tet is ill-conditioned and barycentric coords are unreliable.
_BARY_INV_ABS_MAX = 1e6
# Tolerance for "inside tet" check.  A barycentric coord ≥ -eps is
# considered non-negative (accounts for floating-point rounding).
_BARY_INSIDE_EPS = 1e-4


def _compute_barycentric(
    points: np.ndarray,
    tet_verts: np.ndarray,
    tet_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """For each point, find the containing tet and barycentric coords.

    Points outside all tets are assigned to the **nearest** tet (by
    centroid distance) with clamped barycentric coordinates.

    Args:
        points:    (P, 3)
        tet_verts: (V, 3)
        tet_cells: (T, 4)

    Returns:
        tet_ids:       (P,) int32   — index of the containing tet
        bary:          (P, 4) float32 — barycentric coordinates
        outside_count: int — number of particles that were outside all tets
    """
    P = points.shape[0]
    T = tet_cells.shape[0]

    # Precompute tet inverse matrices  (T, 3, 3)
    v0 = tet_verts[tet_cells[:, 0]]  # (T, 3)
    v1 = tet_verts[tet_cells[:, 1]]
    v2 = tet_verts[tet_cells[:, 2]]
    v3 = tet_verts[tet_cells[:, 3]]

    # Columns of the matrix: (v0-v3, v1-v3, v2-v3)
    mat = np.stack([v0 - v3, v1 - v3, v2 - v3], axis=-1)  # (T, 3, 3)

    # Invert each 3×3 matrix; detect degenerate tets.
    # Near-singular tets (det ≈ 0) produce inv_mat with huge values,
    # which would give extreme barycentric coords.  We mark them so
    # they are **skipped** during the candidate search.
    inv_mat = np.zeros_like(mat)
    tet_is_degenerate = np.zeros(T, dtype=bool)
    degenerate_count = 0
    for t in range(T):
        det_val = np.linalg.det(mat[t])
        if abs(det_val) < _BARY_DET_SINGULAR:
            # Truly singular
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
            # Near-singular: inv has extreme values
            inv_mat[t] = np.eye(3)
            tet_is_degenerate[t] = True
            degenerate_count += 1
        else:
            inv_mat[t] = inv_t
    if degenerate_count > 0:
        print(
            f"[Barycentric] WARNING: {degenerate_count}/{T} degenerate "
            f"(singular/near-singular) tets detected — will be skipped "
            f"as candidates!"
        )

    # Centroids for nearest-tet fallback
    centroids = (v0 + v1 + v2 + v3) / 4.0  # (T, 3)

    tet_ids = np.zeros(P, dtype=np.int32)
    bary = np.zeros((P, 4), dtype=np.float32)
    outside_count = 0

    # Use more candidates for better surface-particle coverage.
    # Surface particles may sit in tets whose centroids are far inside.
    N_CAND = min(128, T)

    # Process in batches for memory efficiency
    BATCH = 4096
    for start in range(0, P, BATCH):
        end = min(start + BATCH, P)
        pts = points[start:end]  # (B, 3)
        B = pts.shape[0]

        # For each point, find closest N_CAND tets by centroid
        dists = np.linalg.norm(
            centroids[None, :, :] - pts[:, None, :], axis=2
        )  # (B, T)
        if N_CAND < T:
            cand_idx = np.argpartition(
                dists, N_CAND, axis=1
            )[:, :N_CAND]  # (B, N_CAND)
        else:
            cand_idx = np.tile(np.arange(T), (B, 1))  # full search

        for i in range(B):
            found = False
            best_tet = -1
            best_min_bary = -np.inf  # track "least outside" tet

            n_cand = cand_idx.shape[1]
            for c in range(n_cand):
                t = cand_idx[i, c]
                # Skip degenerate tets — their inv_mat is unreliable
                if tet_is_degenerate[t]:
                    continue
                p_local = pts[i] - v3[t]
                lam = inv_mat[t] @ p_local  # (3,)
                lam3 = 1.0 - lam[0] - lam[1] - lam[2]
                min_lam = min(lam[0], lam[1], lam[2], lam3)
                # Inside tet if all bary coords >= -eps
                if min_lam >= -_BARY_INSIDE_EPS:
                    tet_ids[start + i] = t
                    bary[start + i] = [lam[0], lam[1], lam[2], lam3]
                    found = True
                    break
                # Track the tet where point is "least outside"
                if min_lam > best_min_bary:
                    best_min_bary = min_lam
                    best_tet = t

            if not found:
                outside_count += 1
                # Fallback: use the tet where point is least outside
                t = best_tet
                if t < 0 or tet_is_degenerate[t]:
                    # All candidates were degenerate; pick nearest
                    # non-degenerate tet by centroid distance.
                    non_degen = np.where(~tet_is_degenerate)[0]
                    if len(non_degen) > 0:
                        cd = np.linalg.norm(
                            centroids[non_degen] - pts[i], axis=1
                        )
                        t = non_degen[np.argmin(cd)]
                    else:
                        t = 0  # last resort
                p_local = pts[i] - v3[t]
                lam = inv_mat[t] @ p_local
                lam3 = 1.0 - lam[0] - lam[1] - lam[2]
                # Clamp to [0, 1] and re-normalize
                raw = np.array(
                    [lam[0], lam[1], lam[2], lam3], dtype=np.float32
                )
                raw = np.maximum(raw, 0.0)
                s = raw.sum()
                if s > 0:
                    raw /= s
                else:
                    raw[:] = 0.25
                tet_ids[start + i] = t
                bary[start + i] = raw

    return tet_ids, bary, outside_count


# ═════════════════════════════════════════════════════════════════════
# Backend
# ═════════════════════════════════════════════════════════════════════

class NewtonVBDBackend(PhysicsBackend):
    """Unified rigid + soft body backend using Newton VBD solver."""

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._builder: Optional[newton.ModelBuilder] = None
        self._model: Optional[newton.Model] = None
        self._solver: Optional[SolverVBD] = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._collision_pipeline = None
        self._contacts = None

        # Per-object info
        self._rigid_bodies: list[_RigidInfo] = []
        self._soft_bodies: list[_SoftInfo] = []
        self._n_particles: int = 0

        # Initial data (kept for get_state)
        self._init_positions: Optional[torch.Tensor] = None
        self._init_covariances: Optional[torch.Tensor] = None

        # Config
        self._gravity = (0.0, 0.0, -9.8)
        self._solver_iterations = 10
        self._collision_geo = "convex_hull"  # for rigid bodies
        self._alpha = None
        self._max_triangles = 500
        self._grid_lim = 2.0

        # Track particle offset for soft bodies
        self._particle_offset = 0

        # Soft body deformation options
        self._debug_soft_no_deformation = False  # skip F update, only interp pos
        self._sv_clamp_min = 0.1   # min allowed singular value of F
        self._sv_clamp_max = 5.0   # max allowed singular value of F
        self._frame_counter = 0    # for periodic diagnostics

        # Self-contact parameters (default: OFF — enabling with wrong
        # radius/margin relative to mesh resolution completely blocks
        # gravity because the solver spends all iterations resolving
        # self-contact instead of falling).
        self._particle_self_contact = False
        self._particle_self_contact_radius = 0.001
        self._particle_self_contact_margin = 0.002

        # Soft contact stiffness (particle↔rigid shape contact).
        # Newton official examples use 1e2 for ground contact.
        # Too high → explosion on impact; too low → penetration.
        self._soft_contact_ke = 1e2
        self._soft_contact_kd = 1e-5
        self._soft_contact_mu = 0.5

        # Max contact buffer size (Newton internal)
        self._rigid_contact_max = 100_000

    # ── PhysicsBackend interface ─────────────────────────────────────

    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        **kwargs,
    ) -> None:
        self._n_particles = positions.shape[0]
        self._init_positions = positions.clone().to(self._device)
        self._init_covariances = covariances.clone().to(self._device)
        self._grid_lim = float(kwargs.get("grid_lim", 2.0))

        iq = kwargs.get("init_quats")
        self._init_quats = iq.clone().to(self._device) if iq is not None else None
        isc = kwargs.get("init_scales")
        self._init_scales = isc.clone().to(self._device) if isc is not None else None

        # Collision geometry for rigid bodies
        self._collision_geo = kwargs.get("collision_geometry", "convex_hull")
        self._alpha = kwargs.get("alpha", None)
        self._max_triangles = int(kwargs.get("max_triangles", 500))

        # Soft body deformation gradient options
        self._debug_soft_no_deformation = bool(
            kwargs.get("debug_soft_no_deformation", False)
        )
        self._sv_clamp_min = float(kwargs.get("sv_clamp_min", 0.1))
        self._sv_clamp_max = float(kwargs.get("sv_clamp_max", 5.0))
        if self._debug_soft_no_deformation:
            print(
                "[NewtonVBD] DEBUG: soft body deformation gradient DISABLED "
                "(positions only, no cov/rot update)"
            )
        else:
            print(
                f"[NewtonVBD] F singular value clamp: "
                f"[{self._sv_clamp_min}, {self._sv_clamp_max}]"
            )

        # Self-contact parameters
        self._particle_self_contact = bool(
            kwargs.get("particle_self_contact", False)
        )
        self._particle_self_contact_radius = float(
            kwargs.get("particle_self_contact_radius", 0.001)
        )
        self._particle_self_contact_margin = float(
            kwargs.get("particle_self_contact_margin", 0.002)
        )
        self._soft_contact_ke = float(
            kwargs.get("soft_contact_ke", 1e2)
        )
        self._soft_contact_kd = float(
            kwargs.get("soft_contact_kd", 1e-5)
        )
        self._soft_contact_mu = float(
            kwargs.get("soft_contact_mu", 0.5)
        )

        # Max contact buffer
        self._rigid_contact_max = int(
            kwargs.get("rigid_contact_max", 100_000)
        )

        # Contact margin
        contact_margin = kwargs.get("contact_margin", 0.01)

        # Create builder
        self._builder = newton.ModelBuilder()
        self._builder.default_shape_cfg.contact_margin = float(contact_margin)

        print(
            f"[NewtonVBD] collision_geometry={self._collision_geo}, "
            f"contact_margin={contact_margin}"
        )

    def set_material(self, material_params: dict) -> None:
        builder = self._builder
        assert builder is not None

        # Gravity
        g = material_params.get("g", [0.0, 0.0, -9.8])
        if isinstance(g, (int, float)):
            g = [0.0, 0.0, -abs(float(g))]
        self._gravity = tuple(g)

        # Solver options
        solver_opts = material_params.get("newton_solver_opts", {})
        self._solver_iterations = solver_opts.get("iterations", 10)

        # Default shape config for rigid bodies
        default_mu = float(material_params.get("mu", 0.5))
        default_density = float(material_params.get("density", 500.0))
        base_cfg = newton.ModelBuilder.ShapeConfig(
            density=default_density,
            mu=default_mu,
        )

        # Create bodies from per-object definitions
        per_object = material_params.get("per_object")
        if per_object is not None:
            for obj in per_object:
                mat = obj.get("material", {})
                physics_type = mat.get("physics", "rigid")

                if physics_type == "rigid":
                    cfg = copy(base_cfg)
                    cfg.density = float(mat.get("density", cfg.density))
                    if "mu" in mat:
                        cfg.mu = float(mat["mu"])
                    self._create_rigid_body(
                        particle_indices=obj["particle_indices"],
                        shape_cfg=cfg,
                        name=obj.get("name", "?"),
                        collision_geo=mat.get("collision_geometry", None),
                    )
                elif physics_type == "soft":
                    self._create_soft_body(
                        particle_indices=obj["particle_indices"],
                        material=mat,
                        name=obj.get("name", "?"),
                    )
                else:
                    raise ValueError(
                        f"Unknown physics type '{physics_type}' for "
                        f"object '{obj.get('name', '?')}'. "
                        "Use 'rigid' or 'soft'."
                    )
        else:
            # Single body → default to rigid
            self._create_rigid_body(
                particle_indices=list(range(self._n_particles)),
                shape_cfg=base_cfg,
                name="single_body",
            )

    def set_boundary_conditions(
        self, bc_params: list, time_params: dict
    ) -> None:
        builder = self._builder
        assert builder is not None

        if not isinstance(bc_params, list):
            return

        for bc in bc_params:
            bc_type = bc.get("type", "")
            if bc_type == "surface_collider":
                normal = bc["normal"]
                point = bc["point"]
                d = -(normal[0] * point[0] + normal[1] * point[1] +
                      normal[2] * point[2])
                surface = bc.get("surface", "slip")
                mu = 0.5
                if surface == "sticky":
                    mu = 1.0
                elif surface == "slip":
                    mu = 0.0
                if "friction" in bc:
                    mu = float(bc["friction"])

                plane_cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
                builder.add_shape_plane(
                    plane=(float(normal[0]), float(normal[1]),
                           float(normal[2]), float(d)),
                    cfg=plane_cfg,
                )
                print(
                    f"[NewtonVBD] Plane: normal={normal}, "
                    f"point={point}, mu={mu}"
                )

    def finalize(self) -> None:
        builder = self._builder
        assert builder is not None

        # VBD requires coloring
        builder.color()

        self._model = builder.finalize(device=self._device)
        self._model.set_gravity(self._gravity)
        print(f"[NewtonVBD] Gravity: {self._gravity}")

        # Soft contact parameters (particle-shape and self-contact)
        self._model.soft_contact_ke = self._soft_contact_ke
        self._model.soft_contact_kd = self._soft_contact_kd
        self._model.soft_contact_mu = self._soft_contact_mu

        # Limit contact buffer
        self._model.rigid_contact_max = self._rigid_contact_max

        # Create VBD solver
        self._solver = SolverVBD(
            self._model,
            iterations=self._solver_iterations,
            particle_enable_self_contact=self._particle_self_contact,
            particle_self_contact_radius=self._particle_self_contact_radius,
            particle_self_contact_margin=self._particle_self_contact_margin,
        )
        print(
            f"[NewtonVBD] SolverVBD: iterations={self._solver_iterations}, "
            f"self_contact={self._particle_self_contact}"
        )
        if self._particle_self_contact:
            print(
                f"[NewtonVBD]   self_contact_radius="
                f"{self._particle_self_contact_radius}, "
                f"margin={self._particle_self_contact_margin}, "
                f"ke={self._soft_contact_ke}"
            )

        # Double-buffered states
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # Collision pipeline
        self._collision_pipeline = (
            newton.CollisionPipelineUnified.from_model(
                self._model,
                broad_phase_mode=newton.BroadPhaseMode.SAP,
            )
        )
        self._contacts = self._model.collide(
            self._state_0,
            collision_pipeline=self._collision_pipeline,
        )

        self._builder = None

    def step(self, dt: float, frame: int) -> None:
        self._state_0.clear_forces()
        self._contacts = self._model.collide(
            self._state_0,
            collision_pipeline=self._collision_pipeline,
        )
        self._solver.step(
            self._state_0,
            self._state_1,
            self._control,
            self._contacts,
            dt,
        )
        self._state_0, self._state_1 = self._state_1, self._state_0

    def get_state(self) -> SimulationState:
        positions = torch.zeros(
            (self._n_particles, 3), device=self._device, dtype=torch.float32
        )
        covariances = torch.zeros(
            (self._n_particles, 6), device=self._device, dtype=torch.float32
        )
        rotations = torch.zeros(
            (self._n_particles, 3, 3), device=self._device, dtype=torch.float32
        )

        has_2dgs = self._init_quats is not None
        if has_2dgs:
            out_quats = torch.zeros(
                (self._n_particles, 4), device=self._device, dtype=torch.float32,
            )
            out_scales = torch.zeros(
                (self._n_particles, self._init_scales.shape[1]),
                device=self._device, dtype=torch.float32,
            )

        # ── Rigid bodies ─────────────────────────────────────────────
        if self._rigid_bodies:
            body_q = self._state_0.body_q.numpy()  # (n_bodies, 7)

            for body in self._rigid_bodies:
                t = body_q[body.body_idx]
                body_pos = torch.tensor(
                    t[:3], device=self._device, dtype=torch.float32
                )
                body_quat = torch.tensor(
                    t[3:7], device=self._device, dtype=torch.float32
                )
                R = _quat_to_rotmat(body_quat)

                idx = body.particle_indices
                local_pos = body.init_local_pos
                new_pos = (R @ local_pos.T).T + body_pos
                positions[idx] = new_pos

                init_cov = body.init_cov_3x3
                new_cov_3x3 = R @ init_cov @ R.T
                covariances[idx] = _pack_cov3x3_to_6(new_cov_3x3)

                R_batch = R.unsqueeze(0).expand(len(idx), -1, -1)
                rotations[idx] = R_batch

                if has_2dgs and body.init_quats is not None:
                    out_quats[idx] = _compose_body_quat_wxyz(body_quat, body.init_quats)
                    out_scales[idx] = body.init_scales

        # ── Soft bodies ──────────────────────────────────────────────
        if self._soft_bodies:
            particle_q = self._state_0.particle_q.numpy()  # (total_verts, 3)
            self._frame_counter += 1
            do_diag = (self._frame_counter <= 3) or (
                self._frame_counter % 50 == 0
            )

            for soft in self._soft_bodies:
                idx = soft.particle_indices
                N = len(idx)

                # Get current tet vertex positions
                off = soft.vert_offset
                cur_verts = particle_q[off:off + soft.vert_count]  # (V, 3)

                # ── Diagnostics (lightweight) ──────────────────────
                if do_diag:
                    z_min = cur_verts[:, 2].min()
                    z_max = cur_verts[:, 2].max()
                    delta = cur_verts - soft.rest_verts
                    max_delta = np.abs(delta).max()
                    mean_dz = delta[:, 2].mean()
                    nan_v = np.isnan(cur_verts).any()
                    inf_v = np.isinf(cur_verts).any()
                    print(
                        f"[VBD-DIAG] frame={self._frame_counter}: "
                        f"z=[{z_min:.4f},{z_max:.4f}], "
                        f"max_delta={max_delta:.6f}, "
                        f"mean_dz={mean_dz:.6f}, "
                        f"NaN={nan_v}, Inf={inf_v}"
                    )

                # Interpolate GS positions from barycentric coords
                tet_idx = soft.tet_ids       # (N,)
                bary = soft.bary_coords      # (N, 4)
                cells = soft.tet_cells       # (T, 4)

                # Gather the 4 tet vertices for each GS particle
                cell_verts = cells[tet_idx]  # (N, 4) — vertex indices
                v0 = cur_verts[cell_verts[:, 0]]  # (N, 3)
                v1 = cur_verts[cell_verts[:, 1]]
                v2 = cur_verts[cell_verts[:, 2]]
                v3 = cur_verts[cell_verts[:, 3]]

                # Barycentric interpolation
                new_pos_np = (
                    bary[:, 0:1] * v0 +
                    bary[:, 1:2] * v1 +
                    bary[:, 2:3] * v2 +
                    bary[:, 3:4] * v3
                )
                positions[idx] = torch.from_numpy(
                    new_pos_np.astype(np.float32)
                ).to(self._device)

                # ── Debug mode: skip deformation gradient ──────────
                if self._debug_soft_no_deformation:
                    # Keep original covariance & identity rotation
                    covariances[idx] = soft.init_cov_6.to(self._device)
                    eye = torch.eye(
                        3, device=self._device, dtype=torch.float32
                    )
                    rotations[idx] = eye.unsqueeze(0).expand(N, -1, -1)
                    continue

                # ── Compute deformation gradient F per-tet ─────────
                rest = soft.rest_verts
                rest_cells = cells[tet_idx]
                r0 = rest[rest_cells[:, 0]]
                r1 = rest[rest_cells[:, 1]]
                r2 = rest[rest_cells[:, 2]]
                r3 = rest[rest_cells[:, 3]]

                # Rest-pose edge matrix columns: (N, 3, 3)
                D_rest = np.stack(
                    [r0 - r3, r1 - r3, r2 - r3], axis=-1
                )
                # Current edge matrix
                D_cur = np.stack(
                    [v0 - v3, v1 - v3, v2 - v3], axis=-1
                )

                # F = D_cur @ inv(D_rest), per unique tet
                F_np = np.zeros((N, 3, 3), dtype=np.float32)
                unique_tets = np.unique(tet_idx)
                inv_cache: dict[int, np.ndarray] = {}
                for ut in unique_tets:
                    t_cells = cells[ut]
                    dr = np.stack([
                        rest[t_cells[0]] - rest[t_cells[3]],
                        rest[t_cells[1]] - rest[t_cells[3]],
                        rest[t_cells[2]] - rest[t_cells[3]],
                    ], axis=-1)
                    try:
                        inv_cache[ut] = np.linalg.inv(dr).astype(
                            np.float32
                        )
                    except np.linalg.LinAlgError:
                        inv_cache[ut] = np.eye(3, dtype=np.float32)

                for i in range(N):
                    F_np[i] = D_cur[i] @ inv_cache[tet_idx[i]]

                F_t = torch.from_numpy(F_np).to(self._device)

                # ── Clamp F via SVD to prevent extreme deformation ─
                U, S, Vh = torch.linalg.svd(F_t)

                # Diagnostics
                if do_diag:
                    s_min = S.min().item()
                    s_max = S.max().item()
                    det_F = torch.det(F_t)
                    n_inv = (det_F < 0).sum().item()
                    n_nan = torch.isnan(F_t).any(dim=(1, 2)).sum().item()
                    print(
                        f"[VBD-DIAG] frame={self._frame_counter}: "
                        f"F sv_range=[{s_min:.4f}, {s_max:.4f}], "
                        f"inverted_tets={n_inv}/{N}, "
                        f"nan_F={n_nan}"
                    )

                # Clamp singular values
                S_clamped = S.clamp(
                    min=self._sv_clamp_min, max=self._sv_clamp_max
                )

                # Reconstruct clamped F
                F_clamped = U @ torch.diag_embed(S_clamped) @ Vh

                # Update covariances: cov' = F_clamped @ cov_init @ F_clamped^T
                init_cov_6 = soft.init_cov_6.to(self._device)
                init_cov_3x3 = _unpack_cov6_to_3x3(init_cov_6)
                new_cov_3x3 = (
                    F_clamped @ init_cov_3x3 @ F_clamped.transpose(1, 2)
                )
                covariances[idx] = _pack_cov3x3_to_6(new_cov_3x3)

                # Extract rotation: R = U @ V^T (from the same SVD)
                R_batch = U @ Vh
                # Ensure proper rotation (det > 0)
                det = torch.det(R_batch)
                mask = det < 0
                if mask.any():
                    U_fix = U[mask].clone()
                    U_fix[:, :, -1] *= -1
                    R_batch[mask] = U_fix @ Vh[mask]
                rotations[idx] = R_batch

        return SimulationState(
            positions=positions,
            covariances=covariances,
            rotations=rotations,
            quats=out_quats if has_2dgs else None,
            scales=out_scales if has_2dgs else None,
        )

    # ── Rigid body creation ──────────────────────────────────────────

    def _create_rigid_body(
        self,
        particle_indices: list[int],
        shape_cfg: newton.ModelBuilder.ShapeConfig,
        name: str,
        collision_geo: str | None = None,
    ) -> None:
        builder = self._builder
        assert builder is not None

        idx_t = torch.tensor(particle_indices, dtype=torch.long)
        pos = self._init_positions[idx_t].float()
        pos_np = pos.detach().cpu().numpy()

        geo = collision_geo or self._collision_geo
        center = pos.mean(dim=0)
        center_np = center.detach().cpu().numpy()

        # ── Collision shape ──────────────────────────────────────────
        if geo in ("obb", "ellipsoid"):
            self._create_rigid_primitive(
                geo, pos_np, idx_t, pos, particle_indices,
                shape_cfg, name,
            )
        else:
            self._create_rigid_mesh(
                geo, pos_np, idx_t, pos, particle_indices,
                shape_cfg, name,
            )

    def _create_rigid_primitive(
        self, geo, pos_np, idx_t, positions, particle_indices,
        shape_cfg, name,
    ):
        builder = self._builder
        if geo == "obb":
            info = fit_obb(pos_np)
            hx, hy, hz = (float(v) for v in info["half_extents"])
        else:
            info = fit_ellipsoid(pos_np)

        center_np = info["center"]
        axes = info["axes"]
        R = axes.T
        quat = _rotmat_to_wp_quat(R)

        body_idx = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(*center_np.astype(float)),
                q=wp.quat(*quat),
            ),
            key=name,
        )

        if geo == "obb":
            builder.add_shape_box(body_idx, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
            desc = f"box({hx:.3f},{hy:.3f},{hz:.3f})"
        else:
            sa = info["semi_axes"]
            a, b, c = (float(v) for v in sa)
            builder.add_shape_ellipsoid(body_idx, a=a, b=b, c=c, cfg=shape_cfg)
            desc = f"ellipsoid({a:.3f},{b:.3f},{c:.3f})"

        # Body-local positions
        R_inv = np.linalg.inv(R).astype(np.float32)
        local_pos_np = (pos_np - center_np) @ R_inv.T
        local_pos = torch.from_numpy(local_pos_np).to(self._device)
        cov6 = self._init_covariances[idx_t]
        cov_3x3 = _unpack_cov6_to_3x3(cov6)

        iq = self._init_quats[idx_t] if self._init_quats is not None else None
        isc = self._init_scales[idx_t] if self._init_scales is not None else None

        self._rigid_bodies.append(_RigidInfo(
            body_idx=body_idx,
            particle_indices=particle_indices,
            init_local_pos=local_pos,
            init_cov_3x3=cov_3x3,
            init_quats=iq,
            init_scales=isc,
        ))
        print(f"[NewtonVBD] Rigid '{name}': {len(particle_indices)} particles, {desc}")

    def _create_rigid_mesh(
        self, geo, pos_np, idx_t, positions, particle_indices,
        shape_cfg, name,
    ):
        builder = self._builder
        center = positions.mean(dim=0)
        center_np = center.detach().cpu().numpy()

        if geo == "alpha_shape" and _HAS_ALPHA_SHAPE:
            try:
                mesh_verts, mesh_faces = compute_alpha_shape(
                    pos_np, alpha=self._alpha,
                    max_triangles=self._max_triangles,
                )
            except Exception as e:
                print(f"[NewtonVBD] Alpha shape failed for '{name}': {e}")
                geo = "convex_hull"

        if geo == "convex_hull":
            try:
                mesh_verts, mesh_faces = compute_convex_hull(pos_np)
            except Exception:
                mesh_verts, mesh_faces = self._bbox_mesh(pos_np)

        local_verts = mesh_verts - center_np
        collision_mesh = newton.Mesh(
            local_verts.astype(np.float32),
            mesh_faces.flatten().astype(np.int32),
        )

        body_idx = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(*center_np.astype(float)),
                q=wp.quat_identity(),
            ),
            key=name,
        )
        builder.add_shape_mesh(body_idx, mesh=collision_mesh, cfg=shape_cfg)

        local_pos = (positions - center).to(self._device)
        cov6 = self._init_covariances[idx_t]
        cov_3x3 = _unpack_cov6_to_3x3(cov6)

        iq = self._init_quats[idx_t] if self._init_quats is not None else None
        isc = self._init_scales[idx_t] if self._init_scales is not None else None

        self._rigid_bodies.append(_RigidInfo(
            body_idx=body_idx,
            particle_indices=particle_indices,
            init_local_pos=local_pos,
            init_cov_3x3=cov_3x3,
            init_quats=iq,
            init_scales=isc,
        ))
        print(
            f"[NewtonVBD] Rigid '{name}': {len(particle_indices)} particles, "
            f"mesh({geo})={mesh_verts.shape[0]}v/{mesh_faces.shape[0]}f"
        )

    @staticmethod
    def _bbox_mesh(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: axis-aligned bounding box mesh with correct normals."""
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        extent = np.maximum(hi - lo, 1e-3)
        lo -= 0.01 * extent
        hi += 0.01 * extent
        verts = np.array([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)
        center = (lo + hi) / 2.0
        for i in range(faces.shape[0]):
            v0, v1, v2 = verts[faces[i]]
            fn = np.cross(v1 - v0, v2 - v0)
            fc = (v0 + v1 + v2) / 3.0
            if np.dot(fn, fc - center) < 0:
                faces[i, 1], faces[i, 2] = faces[i, 2], faces[i, 1]
        return verts, faces

    # ── Soft body creation ───────────────────────────────────────────

    def _create_soft_body(
        self,
        particle_indices: list[int],
        material: dict,
        name: str,
    ) -> None:
        """Create a soft body using Newton's ``add_soft_grid``.

        Instead of the fragile TetGen pipeline (alpha shape → tet mesh →
        degenerate filter), we generate a **uniform regular tet grid**
        covering the GS particle bounding box.  This matches Newton's
        own examples and produces well-conditioned elements that VBD is
        optimized for.

        Configurable per-object material keys:
            cell_size      : float — tet cell size (auto if omitted)
            grid_resolution: int   — cells per longest axis (default 8,
                                     only used when cell_size is omitted)
            grid_padding   : float — padding around bbox (default 0.05)
            density        : float — mass density   (default 1000)
            k_mu           : float — shear modulus   (default 1e5)
            k_lambda       : float — bulk modulus    (default 1e5)
            k_damp         : float — damping coeff   (default 1e-3)
        """
        builder = self._builder
        assert builder is not None

        idx_t = torch.tensor(particle_indices, dtype=torch.long)
        pos = self._init_positions[idx_t].float()
        pos_np = pos.detach().cpu().numpy()

        # ── Bounding box with padding ──────────────────────────────────
        bbox_min = pos_np.min(axis=0)
        bbox_max = pos_np.max(axis=0)
        padding = float(material.get("grid_padding", 0.05))
        bbox_min = bbox_min - padding
        bbox_max = bbox_max + padding
        extent = bbox_max - bbox_min

        # ── Grid parameters ────────────────────────────────────────────
        # cell_size: explicit value, or auto-compute from grid_resolution.
        cell_size = material.get("cell_size", None)
        if cell_size is not None:
            cell_size = float(cell_size)
        else:
            grid_res = int(material.get("grid_resolution", 8))
            cell_size = float(extent.max() / max(grid_res, 1))

        dim_x = max(1, int(np.ceil(extent[0] / cell_size)))
        dim_y = max(1, int(np.ceil(extent[1] / cell_size)))
        dim_z = max(1, int(np.ceil(extent[2] / cell_size)))

        # Recalculate per-axis cell sizes to exactly cover the bbox.
        cell_x = extent[0] / dim_x
        cell_y = extent[1] / dim_y
        cell_z = extent[2] / dim_z

        # ── Material parameters ────────────────────────────────────────
        density = float(material.get("density", 1e3))
        k_mu = float(material.get("k_mu", 1e5))
        k_lambda = float(material.get("k_lambda", 1e5))
        k_damp = float(material.get("k_damp", 1e-3))

        # ── Add regular tet grid via Newton's API ──────────────────────
        vert_offset = self._particle_offset

        builder.add_soft_grid(
            pos=wp.vec3(
                float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
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

        vert_count = (dim_x + 1) * (dim_y + 1) * (dim_z + 1)
        self._particle_offset += vert_count

        # ── Reconstruct grid vertices in numpy ─────────────────────────
        # Exactly mirrors Newton's add_soft_grid vertex generation order
        # (z-major, then y, then x) so indices match particle_q layout.
        grid_verts = np.zeros((vert_count, 3), dtype=np.float64)
        vi = 0
        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    grid_verts[vi] = [
                        x * cell_x + bbox_min[0],
                        y * cell_y + bbox_min[1],
                        z * cell_z + bbox_min[2],
                    ]
                    vi += 1

        # ── Reconstruct tet connectivity ───────────────────────────────
        # Exactly mirrors Newton's 5-tet alternating decomposition.
        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        tet_list = []
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
                        tet_list.extend([
                            [v0, v1, v4, v3],
                            [v2, v3, v6, v1],
                            [v5, v4, v1, v6],
                            [v7, v6, v3, v4],
                            [v4, v1, v6, v3],
                        ])
                    else:
                        tet_list.extend([
                            [v1, v2, v5, v0],
                            [v3, v0, v7, v2],
                            [v4, v7, v0, v5],
                            [v6, v5, v2, v7],
                            [v5, v2, v7, v0],
                        ])

        tet_cells = np.array(tet_list, dtype=np.int32)

        # ── Embed GS particles in tet grid ─────────────────────────────
        print(
            f"[NewtonVBD] Embedding {len(particle_indices)} GS particles "
            f"in {tet_cells.shape[0]} tets..."
        )
        tet_ids, bary, outside_count = _compute_barycentric(
            pos_np, grid_verts, tet_cells
        )

        if outside_count > 0:
            pct = 100.0 * outside_count / len(particle_indices)
            print(
                f"[NewtonVBD] WARNING: {outside_count}/{len(particle_indices)} "
                f"({pct:.1f}%) GS particles outside grid → clamped"
            )
        else:
            print(
                f"[NewtonVBD] All {len(particle_indices)} particles "
                f"inside grid"
            )

        cov6 = self._init_covariances[idx_t]

        self._soft_bodies.append(_SoftInfo(
            particle_indices=particle_indices,
            tet_ids=tet_ids,
            bary_coords=bary,
            vert_offset=vert_offset,
            vert_count=vert_count,
            tet_cells=tet_cells,
            rest_verts=grid_verts.copy(),
            init_cov_6=cov6,
        ))
        print(
            f"[NewtonVBD] Soft '{name}': {len(particle_indices)} GS particles, "
            f"grid {dim_x}x{dim_y}x{dim_z} = {vert_count} verts, "
            f"{tet_cells.shape[0]} tets, "
            f"cell=[{cell_x:.4f},{cell_y:.4f},{cell_z:.4f}], "
            f"density={density}, k_mu={k_mu:.0e}, k_lambda={k_lambda:.0e}, "
            f"k_damp={k_damp:.0e}"
        )
