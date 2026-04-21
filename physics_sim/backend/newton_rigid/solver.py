"""
Newton XPBD rigid-body backend that implements the PhysicsBackend interface.

Each object (or the whole point cloud) is turned into a convex-hull collision
shape attached to a free-floating rigid body.  Newton's XPBD solver handles
gravity, contacts, and friction.  ``get_state()`` broadcasts each body's
rigid transform (R, t) to all of its Gaussians.

Lifecycle (called by the pipeline):
    initialize  → construct ``_Setup`` (builder + particle data)
    set_material → fill setup.bodies + setup.gravity
    set_boundary_conditions → fill setup.plane_equations
    finalize    → drain ``_Setup`` into ``_Runtime`` (model/solver/states);
                  initial_velocity is consumed once here and discarded.
    step/get_state → operate on ``_Runtime`` only

Phase C decomposition (option α): the backend class holds a mutable
``_Setup`` while being configured and, after ``finalize()``, a
``_Runtime`` with only the fields required for stepping / diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    build_body_shape_config,
    iter_body_specs,
)
from physics_sim.backend.newton_rigid.state_export import export_rigid_state
from physics_sim.backend.spec import MaterialSetupSpec
from physics_sim.config.models import (
    BoundaryCondition,
    BoundingBox,
    NewtonRigidConfig,
    TimeConfig,
)
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
    initial_velocity: Optional[tuple[float, float, float]] = None
    init_quats: Optional[torch.Tensor] = None   # (N, 4) wxyz — for 2DGS
    init_scales: Optional[torch.Tensor] = None   # (N, 2|3) — for 2DGS


# ── Setup / Runtime state containers ─────────────────────────────────────


@dataclass
class _Setup:
    """Mutable configuration-phase state (``initialize`` → ``finalize``)."""

    builder: newton.ModelBuilder
    init_positions: torch.Tensor
    init_covariances: torch.Tensor
    init_quats: Optional[torch.Tensor]
    init_scales: Optional[torch.Tensor]
    n_particles: int
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    bodies: list[_BodyInfo] = field(default_factory=list)
    gravity: Optional[tuple[float, float, float]] = None
    plane_equations: list[tuple[list[float], float]] = field(default_factory=list)


@dataclass
class _Runtime:
    """Immutable-layout runtime state produced by ``finalize``."""

    model: "newton.Model"
    solver: SolverXPBD
    state_0: "newton.State"
    state_1: "newton.State"
    control: "newton.Control"
    collision_pipeline: "newton.CollisionPipeline"
    contacts: object
    bodies: list[_BodyInfo]
    plane_equations: list[tuple[list[float], float]]
    n_particles: int
    init_scales: Optional[torch.Tensor]
    has_init_quats: bool
    last_valid_body_q: Optional[np.ndarray] = None


# ── Backend ──────────────────────────────────────────────────────────────


class NewtonRigidBackend(PhysicsBackend):
    """Newton XPBD rigid-body physics backend for 3DGS scenes."""

    def __init__(self, *, cfg: NewtonRigidConfig, device: str = "cuda:0"):
        self._cfg = cfg
        self._device = device
        self._setup: Optional[_Setup] = None
        self._runtime: Optional[_Runtime] = None

        # Cfg-derived constants consumed at setup time.
        self._solver_iterations: int = int(cfg.solver_iterations)
        self._solver_relaxation: float = float(cfg.contact_relaxation)
        self._collision_geo = normalize_collision_geo(cfg.collision_geometry)
        self._use_sdf = bool(cfg.use_sdf)
        self._sdf_resolution = int(cfg.sdf_resolution)
        self._sdf_narrow_band: tuple[float, float] = (-0.01, 0.01)
        self._alpha: float | None = None
        self._max_triangles: int = 300

    # ── Lifecycle helpers ────────────────────────────────────────────

    def _require_setup(self, operation: str) -> _Setup:
        if self._setup is None:
            raise lifecycle_error(
                owner="newton_rigid",
                operation=operation,
                expected="initialize() must run before this operation",
            )
        return self._setup

    def _require_runtime(self, operation: str) -> _Runtime:
        if self._runtime is None:
            raise lifecycle_error(
                owner="newton_rigid",
                operation=operation,
                expected="finalize() must run before this operation",
            )
        return self._runtime

    # ── PhysicsBackend interface ─────────────────────────────────────

    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        *,
        init_quats: Optional[torch.Tensor] = None,
        init_scales: Optional[torch.Tensor] = None,
    ) -> None:
        """Construct ``_Setup`` from initial particle data.

        For rigid body simulation, ``volumes`` is not used (Newton
        computes mass/inertia from the shape + density).
        """
        pos_np = positions.detach().cpu().numpy()
        builder = newton.ModelBuilder()
        contact_margin = 0.01 if self._cfg.contact_margin is None else float(self._cfg.contact_margin)
        builder.default_shape_cfg.contact_margin = contact_margin
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

        self._setup = _Setup(
            builder=builder,
            init_positions=positions.clone().to(self._device),
            init_covariances=covariances.clone().to(self._device),
            init_quats=init_quats.clone().to(self._device) if init_quats is not None else None,
            init_scales=init_scales.clone().to(self._device) if init_scales is not None else None,
            n_particles=int(positions.shape[0]),
            bbox_lo=pos_np.min(axis=0),
            bbox_hi=pos_np.max(axis=0),
        )

    def set_material(self, spec: MaterialSetupSpec) -> None:
        """Create rigid bodies from per-object info and record gravity."""
        setup = self._require_setup("set_material")
        setup.gravity = tuple(float(x) for x in spec.gravity)
        specs = iter_body_specs(per_object=spec.per_object, n_particles=setup.n_particles)
        for body in specs:
            shape_cfg = build_body_shape_config(
                material=body.material,
                use_sdf=self._use_sdf,
                sdf_resolution=self._sdf_resolution,
                sdf_narrow_band=self._sdf_narrow_band,
            )
            self._create_body(
                setup=setup,
                particle_indices=body.particle_indices,
                shape_cfg=shape_cfg,
                name=body.name,
                collision_geo=body.material.collision_geometry,
                initial_velocity=body.initial_velocity,
            )

    def set_boundary_conditions(
        self,
        bcs: list[BoundaryCondition],
        time: TimeConfig,
    ) -> None:
        """Add collision planes for ground / walls."""
        setup = self._require_setup("set_boundary_conditions")
        new_planes = register_boundary_conditions(
            builder=setup.builder,
            bc_params=bcs,
            bbox_lo=setup.bbox_lo,
            bbox_hi=setup.bbox_hi,
        )
        base_idx = len(setup.plane_equations)
        setup.plane_equations.extend(new_planes)
        for idx, (normal, d_value) in enumerate(new_planes, start=base_idx):
            LOGGER.info(
                "[NewtonRigid] Plane #%s: normal=[%.6f, %.6f, %.6f], d=%.6f",
                idx,
                normal[0],
                normal[1],
                normal[2],
                d_value,
            )
        if any(isinstance(bc, BoundingBox) for bc in bcs):
            lo = setup.bbox_lo
            hi = setup.bbox_hi
            LOGGER.info(
                f"[NewtonRigid] Bounding box "
                f"[{lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}] – "
                f"[{hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f}]"
            )

    def finalize(self) -> None:
        """Drain ``_Setup`` into ``_Runtime`` (model/solver/states)."""
        setup = self._require_setup("finalize")
        if setup.gravity is None:
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
        model = setup.builder.finalize(device=self._device)
        model.set_gravity(setup.gravity)
        LOGGER.info("[NewtonRigid] gravity=%s", setup.gravity)

        # Limit contact buffer to prevent excessive memory allocation.
        model.rigid_contact_max = 100000

        # ── Create solver ───────────────────────────────────────────
        solver = SolverXPBD(
            model,
            iterations=self._solver_iterations,
            rigid_contact_relaxation=self._solver_relaxation,
        )
        LOGGER.info(
            f"[NewtonRigid] SolverXPBD: iterations={self._solver_iterations}, "
            f"contact_relaxation={self._solver_relaxation}"
        )

        # ── Create double-buffered states + control ─────────────────
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # ── Consume initial velocities once and clear on bodies ─────
        # body_qd is (n_bodies, 6): [vx, vy, vz, wx, wy, wz]
        if any(b.initial_velocity is not None for b in setup.bodies):
            qd_0 = state_0.body_qd.numpy()
            qd_1 = state_1.body_qd.numpy()
            for body in setup.bodies:
                if body.initial_velocity is not None:
                    vx, vy, vz = body.initial_velocity
                    qd_0[body.body_idx, :3] = [vx, vy, vz]
                    qd_1[body.body_idx, :3] = [vx, vy, vz]
                    LOGGER.info(
                        f"[NewtonRigid] Set initial velocity for body {body.body_idx}: "
                        f"({vx}, {vy}, {vz})"
                    )
                    body.initial_velocity = None  # consumed
            state_0.body_qd.assign(qd_0)
            state_1.body_qd.assign(qd_1)

        # ── Collision pipeline ──────────────────────────────────────
        collision_pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            broad_phase="sap",
        )
        contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

        self._runtime = _Runtime(
            model=model,
            solver=solver,
            state_0=state_0,
            state_1=state_1,
            control=control,
            collision_pipeline=collision_pipeline,
            contacts=contacts,
            bodies=setup.bodies,
            plane_equations=setup.plane_equations,
            n_particles=setup.n_particles,
            init_scales=setup.init_scales,
            has_init_quats=setup.init_quats is not None,
        )
        # Release setup-only data (builder, init_positions, init_covariances).
        self._setup = None

    def step(self, dt: float, frame: int) -> None:
        """Advance the rigid body simulation by one substep."""
        rt = self._require_runtime("step")
        rt.state_0.clear_forces()
        rt.contacts = rt.model.collide(rt.state_0, collision_pipeline=rt.collision_pipeline)
        rt.solver.step(rt.state_0, rt.state_1, rt.control, rt.contacts, dt)
        # Swap states
        rt.state_0, rt.state_1 = rt.state_1, rt.state_0

    def get_diagnostics(self) -> dict:
        """Return per-body position, velocity, and plane distances for debugging."""
        rt = self._require_runtime("get_diagnostics")
        body_q = rt.state_0.body_q.numpy()   # (n_bodies, 7)
        body_qd = rt.state_0.body_qd.numpy()  # (n_bodies, 6)
        diag = {"bodies": []}
        for body in rt.bodies:
            bq = body_q[body.body_idx]
            bqd = body_qd[body.body_idx]
            pos = bq[:3].tolist()
            quat = bq[3:7].tolist()
            vel = bqd[:3].tolist()
            ang_vel = bqd[3:6].tolist()
            plane_dists = []
            for pi, (n, d) in enumerate(rt.plane_equations):
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
        rt = self._require_runtime("get_state")
        state, rt.last_valid_body_q = export_rigid_state(
            state_0=rt.state_0,
            bodies=rt.bodies,
            n_particles=rt.n_particles,
            device=self._device,
            init_scales=rt.init_scales,
            has_init_quats=rt.has_init_quats,
            last_valid_body_q=rt.last_valid_body_q,
        )
        return state

    # ── Internal helpers ─────────────────────────────────────────────

    def _create_body(
        self,
        *,
        setup: _Setup,
        particle_indices: list[int],
        shape_cfg: newton.ModelBuilder.ShapeConfig,
        name: str,
        collision_geo: str | None,
        initial_velocity: tuple[float, float, float] | None,
    ) -> None:
        """Create one rigid body from a subset of particles."""
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
            builder=setup.builder,
            init_positions=setup.init_positions,
            init_covariances=setup.init_covariances,
            init_quats=setup.init_quats,
            init_scales=setup.init_scales,
            device=self._device,
            particle_indices=particle_indices,
            shape_cfg=shape_cfg,
            name=name,
            collision_geo=geo,
            alpha=self._alpha,
            max_triangles=self._max_triangles,
        )
        setup.bodies.append(
            _BodyInfo(
                body_idx=built.body_idx,
                particle_indices=particle_indices,
                init_local_pos=built.init_local_pos,
                init_cov_3x3=built.init_cov_3x3,
                initial_velocity=initial_velocity,
                init_quats=built.init_quats,
                init_scales=built.init_scales,
            )
        )
