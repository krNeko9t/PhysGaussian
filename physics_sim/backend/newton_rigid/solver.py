"""
Newton XPBD rigid-body backend that implements the PhysicsBackend interface.

Each object (or the whole point cloud) is turned into a convex-hull collision
shape attached to a free-floating rigid body.  Newton's XPBD solver handles
gravity, contacts, and friction.  ``get_state()`` broadcasts each body's
rigid transform (R, t) to all of its Gaussians.

Lifecycle (called by the pipeline):
    initialize  → store particles, create ModelBuilder
    set_material → create bodies + shapes per object, set gravity
    set_boundary_conditions → add ground planes / walls
    finalize    → finalize model, create solver + states + collision pipeline
    step        → clear forces, collide, solver step, swap states
    get_state   → apply rigid transforms to original particles → SimulationState
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverXPBD

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.geometry.convex_hull import compute_convex_hull

# Alpha Shape gives a much tighter collision mesh than a convex hull
# for organic shapes (animals, food, etc.).  Falls back to convex hull
# if Open3D is not installed.
try:
    from physics_sim.geometry.alpha_shape import compute_alpha_shape
    _HAS_ALPHA_SHAPE = True
except ImportError:
    _HAS_ALPHA_SHAPE = False


# ── Helpers ──────────────────────────────────────────────────────────────


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert a single quaternion (x, y, z, w) to a 3×3 rotation matrix."""
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


def _unpack_cov6_to_3x3(cov6: torch.Tensor) -> torch.Tensor:
    """(N, 6) upper-triangle → (N, 3, 3) symmetric matrix."""
    N = cov6.shape[0]
    out = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    out[:, 0, 0] = cov6[:, 0]
    out[:, 0, 1] = cov6[:, 1]
    out[:, 0, 2] = cov6[:, 2]
    out[:, 1, 0] = cov6[:, 1]
    out[:, 1, 1] = cov6[:, 3]
    out[:, 1, 2] = cov6[:, 4]
    out[:, 2, 0] = cov6[:, 2]
    out[:, 2, 1] = cov6[:, 4]
    out[:, 2, 2] = cov6[:, 5]
    return out


def _pack_cov3x3_to_6(cov3x3: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) symmetric matrix → (N, 6) upper-triangle."""
    out = torch.zeros(
        (cov3x3.shape[0], 6), device=cov3x3.device, dtype=cov3x3.dtype
    )
    out[:, 0] = cov3x3[:, 0, 0]
    out[:, 1] = cov3x3[:, 0, 1]
    out[:, 2] = cov3x3[:, 0, 2]
    out[:, 3] = cov3x3[:, 1, 1]
    out[:, 4] = cov3x3[:, 1, 2]
    out[:, 5] = cov3x3[:, 2, 2]
    return out


# ── Body bookkeeping ─────────────────────────────────────────────────────


@dataclass
class _BodyInfo:
    """Per-rigid-body data for back-mapping transforms to particles."""

    body_idx: int
    particle_indices: list[int]
    init_local_pos: torch.Tensor  # (N, 3) positions in body-local frame
    init_cov_3x3: torch.Tensor   # (N, 3, 3) initial covariances


# ── Backend ──────────────────────────────────────────────────────────────


class NewtonRigidBackend(PhysicsBackend):
    """Newton XPBD rigid-body physics backend for 3DGS scenes.

    Each object becomes a separate rigid body with a convex-hull
    collision shape.  Gravity, ground planes, and inter-body contacts
    are handled by Newton's XPBD solver.
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._builder: Optional[newton.ModelBuilder] = None
        self._model = None
        self._solver = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._collision_pipeline = None
        self._contacts = None

        # Particle data
        self._init_positions: Optional[torch.Tensor] = None
        self._init_covariances: Optional[torch.Tensor] = None
        self._n_particles: int = 0

        # Per-body info
        self._bodies: list[_BodyInfo] = []

        # Deferred configuration
        self._gravity: tuple = (0.0, 0.0, -9.8)
        self._solver_iterations: int = 10

    # ── PhysicsBackend interface ─────────────────────────────────────

    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        *,
        n_grid: int = 100,
        grid_lim: float = 2.0,
        **kwargs,
    ) -> None:
        """Store initial particle data and create ModelBuilder.

        For rigid body simulation, *volumes* and *n_grid* are not used
        (Newton computes mass/inertia from the shape + density).
        """
        self._init_positions = positions.clone().to(self._device)
        self._init_covariances = covariances.clone().to(self._device)
        self._n_particles = positions.shape[0]
        self._grid_lim = grid_lim

        # ── Collision geometry config ───────────────────────────────
        # "alpha_shape" (default, tighter mesh) or "convex_hull" (fallback)
        self._collision_geo = kwargs.get("collision_geometry", "alpha_shape")
        # Alpha parameter: None = auto-estimate from bounding box
        self._alpha = kwargs.get("alpha", None)
        if self._collision_geo == "alpha_shape" and not _HAS_ALPHA_SHAPE:
            print(
                "[NewtonRigid] WARNING: Open3D not available, "
                "falling back to convex_hull"
            )
            self._collision_geo = "convex_hull"

        # Create ModelBuilder (bodies & shapes added in set_material)
        self._builder = newton.ModelBuilder()
        # Contact margin: start detecting contacts before actual penetration.
        # A small positive margin prevents deep interpenetration that causes
        # sudden explosive correction forces (Newton example_sdf uses 0.01).
        contact_margin = kwargs.get("contact_margin", 0.01)
        self._builder.default_shape_cfg.contact_margin = contact_margin
        print(
            f"[NewtonRigid] collision_geometry={self._collision_geo}, "
            f"alpha={self._alpha}, contact_margin={contact_margin}"
        )

    def set_material(self, material_params: dict) -> None:
        """Create rigid bodies from per-object info (or single body).

        Also configures gravity and solver parameters.
        """
        builder = self._builder
        assert builder is not None, "initialize() must be called first"

        # ── Gravity ─────────────────────────────────────────────────
        g = material_params.get("g", [0.0, 0.0, -9.8])
        if isinstance(g, (int, float)):
            g = [0.0, 0.0, -abs(float(g))]
        self._gravity = tuple(g)

        # ── Solver options ──────────────────────────────────────────
        solver_opts = material_params.get("newton_solver_opts", {})
        self._solver_iterations = solver_opts.get("iterations", 10)
        # Relaxation < 1.0 avoids over-correction at contacts (Newton
        # example_sdf uses 0.8; 1.0 = no damping → can oscillate/explode).
        self._solver_relaxation = float(solver_opts.get("contact_relaxation", 0.8))

        # ── Contact parameters ────────────────────────────────────────
        # For XPBD, Newton's default ke/kd work best — only override if
        # the user explicitly sets them in the config.
        default_mu = float(material_params.get("mu", 0.5))
        default_density = float(material_params.get("density", 1000.0))

        # ── Default shape config ────────────────────────────────────
        # Start from Newton's built-in defaults, then layer on config.
        base_cfg = newton.ModelBuilder.ShapeConfig(
            density=default_density,
            mu=default_mu,
        )
        # Only override ke/kd if explicitly provided (not with arbitrary
        # defaults — Newton's own defaults are tuned for XPBD stability).
        if "ke" in material_params:
            base_cfg.ke = float(material_params["ke"])
        if "kd" in material_params:
            base_cfg.kd = float(material_params["kd"])

        # ── Create bodies ───────────────────────────────────────────
        per_object = material_params.get("per_object")

        if per_object is not None:
            for obj in per_object:
                mat = obj.get("material", {})
                cfg = copy(base_cfg)
                cfg.density = float(mat.get("density", cfg.density))
                if "mu" in mat:
                    cfg.mu = float(mat["mu"])
                elif "friction" in mat:
                    cfg.mu = float(mat["friction"])
                if "ke" in mat:
                    cfg.ke = float(mat["ke"])
                if "kd" in mat:
                    cfg.kd = float(mat["kd"])
                self._create_body(
                    particle_indices=obj["particle_indices"],
                    shape_cfg=cfg,
                    name=obj.get("name", "?"),
                )
        else:
            # Single body: all particles
            self._create_body(
                particle_indices=list(range(self._n_particles)),
                shape_cfg=base_cfg,
                name="single_body",
            )

    def set_boundary_conditions(
        self, bc_params: list, time_params: dict
    ) -> None:
        """Add collision planes for ground / walls."""
        builder = self._builder
        assert builder is not None

        if not isinstance(bc_params, list):
            return

        for bc in bc_params:
            bc_type = bc.get("type", "")

            if bc_type == "surface_collider":
                normal = bc["normal"]
                point = bc["point"]
                # Plane equation: n·x + d = 0 → d = -n·point
                d = -(
                    normal[0] * point[0]
                    + normal[1] * point[1]
                    + normal[2] * point[2]
                )
                # Map surface type to friction
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
                    plane=(
                        float(normal[0]),
                        float(normal[1]),
                        float(normal[2]),
                        float(d),
                    ),
                    cfg=plane_cfg,
                )
                print(
                    f"[NewtonRigid] Plane: normal={normal}, "
                    f"point={point}, mu={mu}"
                )

            elif bc_type == "bounding_box":
                # Add 6 planes for [0, grid_lim]^3 bounding box
                lim = self._grid_lim
                margin = 0.01  # slight inset to avoid edge cases
                wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
                # -x, +x, -y, +y, -z, +z
                planes = [
                    (1.0, 0.0, 0.0, -margin),          # x > 0
                    (-1.0, 0.0, 0.0, lim - margin),    # x < lim
                    (0.0, 1.0, 0.0, -margin),           # y > 0
                    (0.0, -1.0, 0.0, lim - margin),     # y < lim
                    (0.0, 0.0, 1.0, -margin),           # z > 0
                    (0.0, 0.0, -1.0, lim - margin),     # z < lim
                ]
                for p in planes:
                    builder.add_shape_plane(plane=p, cfg=wall_cfg)
                print(f"[NewtonRigid] Bounding box [0, {lim}]^3")

    def finalize(self) -> None:
        """Finalize the Newton model and create solver + states."""
        builder = self._builder
        assert builder is not None

        # ── Finalize model ──────────────────────────────────────────
        self._model = builder.finalize(device=self._device)
        self._model.set_gravity(self._gravity)
        print(f"[NewtonRigid] Gravity: {self._gravity}")

        # Limit contact buffer to prevent excessive memory allocation
        self._model.rigid_contact_max = 100000

        # ── Create solver ───────────────────────────────────────────
        # rigid_contact_relaxation < 1.0 prevents over-correction and
        # oscillation at contacts (Newton's example_sdf uses 0.8).
        relaxation = self._solver_relaxation
        self._solver = SolverXPBD(
            self._model,
            iterations=self._solver_iterations,
            rigid_contact_relaxation=relaxation,
        )
        print(
            f"[NewtonRigid] SolverXPBD: iterations={self._solver_iterations}, "
            f"contact_relaxation={relaxation}"
        )

        # ── Create double-buffered states + control ─────────────────
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # ── Collision pipeline ──────────────────────────────────────
        self._collision_pipeline = (
            newton.CollisionPipelineUnified.from_model(
                self._model,
                reduce_contacts=True,
                broad_phase_mode=newton.BroadPhaseMode.SAP,
            )
        )
        self._contacts = self._model.collide(
            self._state_0,
            collision_pipeline=self._collision_pipeline,
        )

        # Release the builder (no longer needed)
        self._builder = None

    def step(self, dt: float, frame: int) -> None:
        """Advance the rigid body simulation by one substep."""
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
        # Swap states
        self._state_0, self._state_1 = self._state_1, self._state_0

    def get_state(self) -> SimulationState:
        """Broadcast rigid-body transforms to all particles.

        For each body:
          new_pos[i] = R @ local_pos[i] + body_position
          new_cov[i] = R @ init_cov[i] @ R^T
          rotation[i] = R  (same for all particles in the body)
        """
        body_q = self._state_0.body_q.numpy()  # (n_bodies, 7)

        # Early NaN detection — if any body has NaN in its pose, the
        # simulation has blown up (usually from interpenetration).
        if np.any(np.isnan(body_q)):
            bad = [
                i for i in range(body_q.shape[0])
                if np.any(np.isnan(body_q[i]))
            ]
            print(
                f"[NewtonRigid] WARNING: NaN detected in body poses "
                f"(bodies {bad}). Simulation may have diverged."
            )

        positions = torch.zeros(
            (self._n_particles, 3),
            device=self._device,
            dtype=torch.float32,
        )
        covariances = torch.zeros(
            (self._n_particles, 6),
            device=self._device,
            dtype=torch.float32,
        )
        rotations = torch.zeros(
            (self._n_particles, 3, 3),
            device=self._device,
            dtype=torch.float32,
        )

        for body in self._bodies:
            t = body_q[body.body_idx]
            body_pos = torch.tensor(
                t[:3], device=self._device, dtype=torch.float32
            )
            body_quat = torch.tensor(
                t[3:7], device=self._device, dtype=torch.float32
            )
            R = _quat_to_rotmat(body_quat)  # (3, 3)

            idx = body.particle_indices
            local_pos = body.init_local_pos  # (N, 3)
            init_cov = body.init_cov_3x3     # (N, 3, 3)

            # new positions = R @ local_pos^T + body_pos
            new_pos = (R @ local_pos.T).T + body_pos  # (N, 3)

            # new covariances = R @ init_cov @ R^T
            R_batch = R.unsqueeze(0)  # (1, 3, 3)
            new_cov_3x3 = R_batch @ init_cov @ R_batch.transpose(-1, -2)
            new_cov_6 = _pack_cov3x3_to_6(new_cov_3x3)  # (N, 6)

            # All particles in this body share the same rotation
            R_expand = R.unsqueeze(0).expand(len(idx), -1, -1)  # (N,3,3)

            positions[idx] = new_pos
            covariances[idx] = new_cov_6
            rotations[idx] = R_expand

        return SimulationState(
            positions=positions,
            covariances=covariances,
            rotations=rotations,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _create_body(
        self,
        particle_indices: list[int],
        shape_cfg: newton.ModelBuilder.ShapeConfig,
        name: str = "body",
    ) -> None:
        """Create one rigid body from a subset of particles."""
        builder = self._builder
        assert builder is not None

        if len(particle_indices) == 0:
            print(f"[NewtonRigid] Skipping empty body '{name}'")
            return

        # ── Gather particle positions ────────────────────────────────
        idx_t = torch.tensor(particle_indices, dtype=torch.long)
        positions = self._init_positions[idx_t]  # (N, 3)
        center = positions.mean(dim=0)  # (3,)

        # ── Compute collision mesh ────────────────────────────────────
        pos_np = positions.detach().cpu().numpy()
        mesh_verts: np.ndarray
        mesh_faces: np.ndarray

        if self._collision_geo == "alpha_shape":
            try:
                mesh_verts, mesh_faces = compute_alpha_shape(
                    pos_np, alpha=self._alpha
                )
            except Exception as e:
                print(
                    f"[NewtonRigid] WARNING: alpha shape failed for "
                    f"'{name}': {e}. Falling back to convex hull."
                )
                try:
                    mesh_verts, mesh_faces = compute_convex_hull(pos_np)
                except Exception as e2:
                    print(
                        f"[NewtonRigid] WARNING: convex hull also failed: "
                        f"{e2}. Using bounding box."
                    )
                    mesh_verts, mesh_faces = self._bbox_mesh(pos_np)
        else:
            # convex_hull mode
            try:
                mesh_verts, mesh_faces = compute_convex_hull(pos_np)
            except Exception as e:
                print(
                    f"[NewtonRigid] WARNING: convex hull failed for "
                    f"'{name}': {e}. Using bounding box fallback."
                )
                mesh_verts, mesh_faces = self._bbox_mesh(pos_np)

        # Mesh vertices in body-local frame (centered on body origin)
        center_np = center.detach().cpu().numpy()
        local_verts = mesh_verts - center_np
        collision_mesh = newton.Mesh(
            local_verts.astype(np.float32),
            mesh_faces.flatten().astype(np.int32),
        )

        # ── Add body + shape to ModelBuilder ─────────────────────────
        body_idx = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(
                    float(center_np[0]),
                    float(center_np[1]),
                    float(center_np[2]),
                ),
                q=wp.quat_identity(),
            ),
            key=name,
        )
        builder.add_shape_mesh(body_idx, mesh=collision_mesh, cfg=shape_cfg)

        # ── Store back-mapping data ──────────────────────────────────
        local_pos = (positions - center).to(self._device)  # (N, 3)
        cov6 = self._init_covariances[idx_t]  # (N, 6)
        cov_3x3 = _unpack_cov6_to_3x3(cov6)  # (N, 3, 3)

        self._bodies.append(
            _BodyInfo(
                body_idx=body_idx,
                particle_indices=particle_indices,
                init_local_pos=local_pos,
                init_cov_3x3=cov_3x3,
            )
        )
        print(
            f"[NewtonRigid] Body '{name}': {len(particle_indices)} particles, "
            f"mesh({self._collision_geo})={mesh_verts.shape[0]} verts / "
            f"{mesh_faces.shape[0]} faces, density={shape_cfg.density}"
        )

    @staticmethod
    def _bbox_mesh(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: axis-aligned bounding box as 12-triangle mesh."""
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        # Slight expansion to avoid degenerate box
        extent = hi - lo
        extent = np.maximum(extent, 1e-3)
        lo = lo - 0.01 * extent
        hi = hi + 0.01 * extent

        verts = np.array(
            [
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]],
                [lo[0], hi[1], hi[2]],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],  # front
                [2, 3, 7], [2, 7, 6],  # back
                [0, 4, 7], [0, 7, 3],  # left
                [1, 2, 6], [1, 6, 5],  # right
            ],
            dtype=np.int32,
        )
        return verts, faces
