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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import newton
from newton.solvers import SolverXPBD

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.newton_rigid.boundary_conditions import register_boundary_conditions
from physics_sim.backend.newton_rigid.collider_builders import (
    create_rigid_body,
    normalize_collision_geo,
)
from physics_sim.backend.newton_rigid.materials import (
    build_base_shape_config,
    build_shape_config_for_body,
    iter_body_specs,
    resolve_gravity,
    resolve_solver_options,
)
from physics_sim.backend.newton_rigid.state_export import export_rigid_state
from physics_sim.errors import configuration_error, lifecycle_error
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


# ── Body bookkeeping ─────────────────────────────────────────────────────


@dataclass
class _BodyInfo:
    """Per-rigid-body data for back-mapping transforms to particles."""

    body_idx: int
    particle_indices: list[int]
    init_local_pos: torch.Tensor  # (N, 3) positions in body-local frame
    init_cov_3x3: torch.Tensor   # (N, 3, 3) initial covariances
    initial_velocity: Optional[tuple[float, float, float]] = None  # linear velocity [vx, vy, vz]
    init_quats: Optional[torch.Tensor] = None   # (N, 4) wxyz — for 2DGS
    init_scales: Optional[torch.Tensor] = None   # (N, 2|3) — for 2DGS


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

        # Freeze-on-divergence: last valid body poses
        self._last_valid_body_q: Optional[np.ndarray] = None

        # Deferred configuration
        self._gravity: tuple[float, float, float] | None = None
        self._solver_iterations: int = 10
        self._solver_relaxation: float = 0.8

        # Recorded plane equations for diagnostics: list of (normal_3, d)
        self._plane_equations: list[tuple[list[float], float]] = []
        self._bbox_lo: np.ndarray | None = None
        self._bbox_hi: np.ndarray | None = None

    # ── PhysicsBackend interface ─────────────────────────────────────

    def _require_builder(self, operation: str) -> newton.ModelBuilder:
        builder = self._builder
        if builder is None:
            raise lifecycle_error(
                owner="newton_rigid",
                operation=operation,
                expected="initialize() must run before this operation",
            )
        return builder

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
        # Compute bounding box from actual particle positions (not grid_lim)
        pos_np = positions.detach().cpu().numpy()
        self._bbox_lo = pos_np.min(axis=0)
        self._bbox_hi = pos_np.max(axis=0)

        # 2DGS support: store per-particle quats & scales for direct output
        iq = kwargs.get("init_quats")
        self._init_quats = iq.clone().to(self._device) if iq is not None else None
        isc = kwargs.get("init_scales")
        self._init_scales = isc.clone().to(self._device) if isc is not None else None

        # ── Collision geometry config ───────────────────────────────
        # Supported: "obb", "ellipsoid", "convex_hull", "alpha_shape"
        raw_collision_geo = kwargs.get("collision_geometry", "obb")
        self._collision_geo = normalize_collision_geo(raw_collision_geo)
        # Alpha shape specific: alpha parameter and max triangle count
        self._alpha = kwargs.get("alpha", None)
        self._max_triangles = int(kwargs.get("max_triangles", 300))
        # SDF collision: Newton auto-generates an SDF from the mesh and
        # uses distance-field queries instead of triangle intersection.
        # This makes mesh quality (non-manifold, non-watertight) far less
        # critical and is the approach used in Newton's own example_sdf.
        self._use_sdf = bool(kwargs.get("use_sdf", False))
        self._sdf_resolution = int(kwargs.get("sdf_resolution", 64))
        self._sdf_narrow_band = kwargs.get(
            "sdf_narrow_band", (-0.01, 0.01)
        )
        # Create ModelBuilder (bodies & shapes added in set_material)
        self._builder = newton.ModelBuilder()
        # Contact margin: start detecting contacts before actual penetration.
        # A small positive margin prevents deep interpenetration that causes
        # sudden explosive correction forces (Newton example_sdf uses 0.01).
        contact_margin = kwargs.get("contact_margin", 0.01)
        self._builder.default_shape_cfg.contact_margin = contact_margin
        sdf_str = (
            f", use_sdf=True (res={self._sdf_resolution})"
            if self._use_sdf else ""
        )
        LOGGER.info(
            "[NewtonRigid] collision_geometry=%s%s contact_margin=%s",
            self._collision_geo,
            sdf_str,
            contact_margin,
        )

    def set_material(self, material_params: dict) -> None:
        """Create rigid bodies from per-object info (or single body).

        Also configures gravity and solver parameters.
        """
        self._require_builder("set_material")

        self._gravity = resolve_gravity(material_params)
        self._solver_iterations, self._solver_relaxation = resolve_solver_options(material_params)
        base_cfg = build_base_shape_config(
            material_params=material_params,
            use_sdf=self._use_sdf,
            sdf_resolution=self._sdf_resolution,
            sdf_narrow_band=self._sdf_narrow_band,
        )
        specs = iter_body_specs(material_params, n_particles=self._n_particles)
        for spec in specs:
            cfg = build_shape_config_for_body(base_cfg=base_cfg, body_material=spec.material)
            self._create_body(
                particle_indices=spec.particle_indices,
                shape_cfg=cfg,
                name=spec.name,
                collision_geo=spec.material.get("collision_geometry"),
                initial_velocity=spec.material.get("initial_velocity"),
            )

    def set_boundary_conditions(
        self, bc_params: list, time_params: dict
    ) -> None:
        """Add collision planes for ground / walls."""
        builder = self._require_builder("set_boundary_conditions")
        new_planes = register_boundary_conditions(
            builder=builder,
            bc_params=bc_params,
            bbox_lo=self._bbox_lo,
            bbox_hi=self._bbox_hi,
        )
        base_idx = len(self._plane_equations)
        self._plane_equations.extend(new_planes)
        for idx, (normal, d_value) in enumerate(new_planes, start=base_idx):
            LOGGER.info(
                "[NewtonRigid] Plane #%s: normal=[%.6f, %.6f, %.6f], d=%.6f",
                idx,
                normal[0],
                normal[1],
                normal[2],
                d_value,
            )
        if any(isinstance(bc, dict) and bc.get("type") == "bounding_box" for bc in bc_params):
            lo = self._bbox_lo
            hi = self._bbox_hi
            if lo is not None and hi is not None:
                LOGGER.info(
                    f"[NewtonRigid] Bounding box "
                    f"[{lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}] – "
                    f"[{hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f}]"
                )

    def finalize(self) -> None:
        """Finalize the Newton model and create solver + states."""
        builder = self._require_builder("finalize")
        if self._gravity is None:
            detail = "gravity missing; call set_material() before finalize()"
            LOGGER.error(
                "[NewtonRigid] backend=newton_rigid operation=finalize detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_rigid",
                operation="finalize",
                expected="set_material() must set gravity before finalize()",
                detail=detail,
            )

        # ── Finalize model ──────────────────────────────────────────
        self._model = builder.finalize(device=self._device)
        self._model.set_gravity(self._gravity)
        LOGGER.info("[NewtonRigid] gravity=%s", self._gravity)

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
        LOGGER.info(
            f"[NewtonRigid] SolverXPBD: iterations={self._solver_iterations}, "
            f"contact_relaxation={relaxation}"
        )

        # ── Create double-buffered states + control ─────────────────
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # ── Apply initial velocities ──────────────────────────────────
        # body_qd is (n_bodies, 6): [vx, vy, vz, wx, wy, wz]
        # Note: .numpy() returns a copy, so we must use .assign() to write back
        has_initial_vel = any(b.initial_velocity is not None for b in self._bodies)
        if has_initial_vel:
            qd_0 = self._state_0.body_qd.numpy()
            qd_1 = self._state_1.body_qd.numpy()
            for body in self._bodies:
                if body.initial_velocity is not None:
                    vx, vy, vz = body.initial_velocity
                    qd_0[body.body_idx, :3] = [vx, vy, vz]
                    qd_1[body.body_idx, :3] = [vx, vy, vz]
                    LOGGER.info(
                        f"[NewtonRigid] Set initial velocity for body {body.body_idx}: "
                        f"({vx}, {vy}, {vz})"
                    )
            self._state_0.body_qd.assign(qd_0)
            self._state_1.body_qd.assign(qd_1)

        # ── Collision pipeline ──────────────────────────────────────
        # Newton ≥1.1: unified pipeline is `CollisionPipeline`; broad phase is a string.
        self._collision_pipeline = newton.CollisionPipeline(
            self._model,
            reduce_contacts=True,
            broad_phase="sap",
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

    def get_diagnostics(self) -> dict:
        """Return per-body position, velocity, and plane distances for debugging."""
        body_q = self._state_0.body_q.numpy()   # (n_bodies, 7)
        body_qd = self._state_0.body_qd.numpy()  # (n_bodies, 6)
        diag = {"bodies": []}
        for body in self._bodies:
            bq = body_q[body.body_idx]
            bqd = body_qd[body.body_idx]
            pos = bq[:3].tolist()
            quat = bq[3:7].tolist()
            vel = bqd[:3].tolist()
            ang_vel = bqd[3:6].tolist()
            plane_dists = []
            for pi, (n, d) in enumerate(self._plane_equations):
                signed_dist = n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] + d
                plane_dists.append((pi, signed_dist))
            diag["bodies"].append(dict(
                name=f"body_{body.body_idx}",
                pos=pos, quat=quat, vel=vel, ang_vel=ang_vel,
                plane_distances=plane_dists,
                has_nan=bool(np.any(~np.isfinite(bq))),
            ))
        return diag

    def get_state(self) -> SimulationState:
        state, self._last_valid_body_q = export_rigid_state(
            state_0=self._state_0,
            bodies=self._bodies,
            n_particles=self._n_particles,
            device=self._device,
            init_scales=self._init_scales,
            has_init_quats=self._init_quats is not None,
            last_valid_body_q=self._last_valid_body_q,
        )
        return state

    # ── Internal helpers ─────────────────────────────────────────────

    def _create_body(
        self,
        particle_indices: list[int],
        shape_cfg: newton.ModelBuilder.ShapeConfig,
        name: str = "body",
        collision_geo: str | None = None,
        initial_velocity: list | tuple | None = None,
    ) -> None:
        """Create one rigid body from a subset of particles.

        ``collision_geo`` overrides the global ``self._collision_geo``
        for this particular body (enables per-object config).

        ``initial_velocity`` is [vx, vy, vz] in MPM space (applied in finalize).
        """
        init_vel_tuple = None
        if initial_velocity is not None:
            if not isinstance(initial_velocity, (list, tuple)):
                detail = (
                    f"body={name} initial_velocity_type="
                    f"{type(initial_velocity).__name__}"
                )
                LOGGER.error(
                    "[NewtonRigid] backend=newton_rigid operation=_create_body "
                    "detail=%s",
                    detail,
                )
                raise configuration_error(
                    owner="newton_rigid",
                    operation="_create_body",
                    expected="initial_velocity must be [vx, vy, vz]",
                    detail=detail,
                )
            if len(initial_velocity) != 3:
                detail = (
                    f"body={name} initial_velocity={initial_velocity!r} "
                    "must have 3 values"
                )
                LOGGER.error(
                    "[NewtonRigid] backend=newton_rigid operation=_create_body "
                    "detail=%s",
                    detail,
                )
                raise configuration_error(
                    owner="newton_rigid",
                    operation="_create_body",
                    expected="initial_velocity must be [vx, vy, vz]",
                    detail=detail,
                )
            init_vel_tuple = tuple(float(v) for v in initial_velocity)
        builder = self._require_builder("_create_body")
        if len(particle_indices) == 0:
            detail = f"body={name} particle_count=0"
            LOGGER.error(
                "[NewtonRigid] backend=newton_rigid operation=_create_body detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_rigid",
                operation="_create_body",
                expected="body must contain at least one particle",
                detail=detail,
            )

        geo = normalize_collision_geo(collision_geo or self._collision_geo)
        built = create_rigid_body(
            builder=builder,
            init_positions=self._init_positions,
            init_covariances=self._init_covariances,
            init_quats=self._init_quats,
            init_scales=self._init_scales,
            device=self._device,
            particle_indices=particle_indices,
            shape_cfg=shape_cfg,
            name=name,
            collision_geo=geo,
            alpha=self._alpha,
            max_triangles=self._max_triangles,
        )
        self._bodies.append(
            _BodyInfo(
                body_idx=built.body_idx,
                particle_indices=particle_indices,
                init_local_pos=built.init_local_pos,
                init_cov_3x3=built.init_cov_3x3,
                initial_velocity=init_vel_tuple,
                init_quats=built.init_quats,
                init_scales=built.init_scales,
            )
        )
