"""
Newton VBD backend — unified rigid + soft body simulation.

Uses a **single** ``ModelBuilder`` and ``SolverVBD`` to handle both
rigid bodies (``add_body`` + ``add_shape_*``) and FEM soft bodies
(``add_soft_grid``) in one scene, with automatic contact handling.

Each object in the config declares ``physics="rigid"`` (``VBDRigidBody``)
or ``physics="soft"`` (``VBDSoftBody``).  Rigid bodies reuse the same
collision-geometry strategies as ``NewtonRigidBackend``.  Soft bodies
use a regular tetrahedral grid (Newton's ``add_soft_grid``) covering the
GS particle bounding box.

Lifecycle (called by the pipeline):
    initialize  → construct ``_Setup`` (builder + particle data)
    set_material → fill setup rigid/soft bodies + gravity
    set_boundary_conditions → register planes on builder
    finalize    → drain ``_Setup`` into ``_Runtime``
    step/get_state → operate on ``_Runtime`` only

Phase C decomposition (option α): setup-phase data lives on ``_Setup``;
``_Runtime`` holds only the fields needed during stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverVBD

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.newton_common.boundary import surface_plane_from_bc
from physics_sim.backend.newton_common.pin_kernel import (
    PinToBodyHook,
    launch_pin_hook,
)
from physics_sim.backend.newton_vbd.rigid_mesh import create_rigid_body
from physics_sim.backend.newton_vbd.soft_grid import build_soft_grid_embedding
from physics_sim.backend.newton_vbd.state_export import export_state
from physics_sim.backend.spec import MaterialSetupSpec
from physics_sim.config.models import (
    BoundaryCondition,
    NewtonVBDConfig,
    SurfaceCollider,
    TimeConfig,
    VBDMaterial,
    VBDRigidBody,
    VBDSoftBody,
)
from physics_sim.errors import configuration_error, lifecycle_error
from physics_sim.logging_utils import get_logger
from physics_sim.scene.constraint_resolver import (
    ResolvedCollideOnly,
    ResolvedConstraint,
    ResolvedPinToBody,
    ResolvedPinToWorld,
)

LOGGER = get_logger(__name__)


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
    name: str = ""
    kinematic: bool = False


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
    name: str = ""


# ── Setup / Runtime state containers ─────────────────────────────────


@dataclass
class _Setup:
    """Mutable configuration-phase state (``initialize`` → ``finalize``)."""

    builder: newton.ModelBuilder
    init_positions: torch.Tensor
    init_covariances: torch.Tensor
    init_quats: Optional[torch.Tensor]
    init_scales: Optional[torch.Tensor]
    n_particles: int
    rigid_bodies: list[_RigidInfo] = field(default_factory=list)
    soft_bodies: list[_SoftInfo] = field(default_factory=list)
    particle_offset: int = 0
    gravity: Optional[tuple[float, float, float]] = None


@dataclass
class _Runtime:
    """Runtime-phase state produced by ``finalize``."""

    model: "newton.Model"
    solver: SolverVBD
    state_0: "newton.State"
    state_1: "newton.State"
    control: "newton.Control"
    collision_pipeline: "newton.CollisionPipeline"
    contacts: object
    rigid_bodies: list[_RigidInfo]
    soft_bodies: list[_SoftInfo]
    n_particles: int
    init_quats: Optional[torch.Tensor]
    init_scales: Optional[torch.Tensor]
    frame_counter: int = 0
    pin_to_body_hooks: list[PinToBodyHook] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════
# Backend
# ═════════════════════════════════════════════════════════════════════

class NewtonVBDBackend(PhysicsBackend):
    """Unified rigid + soft body backend using Newton VBD solver."""

    def __init__(self, *, cfg: NewtonVBDConfig, device: str = "cuda:0"):
        self._cfg = cfg
        self._device = device
        self._setup: Optional[_Setup] = None
        self._runtime: Optional[_Runtime] = None

        # Cfg-derived constants consumed at setup/finalize time.
        self._solver_iterations = int(cfg.solver_iterations)
        self._collision_geo = cfg.collision_geometry
        self._use_sdf = bool(cfg.use_sdf)
        self._alpha: float | None = None
        self._max_triangles = 500
        self._grid_lim = float(cfg.grid_lim)

        self._debug_soft_no_deformation = bool(cfg.debug_soft_no_deformation)
        self._sv_clamp_min = 0.1 if cfg.sv_clamp_min is None else float(cfg.sv_clamp_min)
        self._sv_clamp_max = 5.0 if cfg.sv_clamp_max is None else float(cfg.sv_clamp_max)

        self._particle_self_contact = (
            False if cfg.particle_self_contact is None
            else bool(cfg.particle_self_contact)
        )
        self._particle_self_contact_radius = (
            0.001 if cfg.particle_self_contact_radius is None
            else float(cfg.particle_self_contact_radius)
        )
        self._particle_self_contact_margin = (
            0.002 if cfg.particle_self_contact_margin is None
            else float(cfg.particle_self_contact_margin)
        )

        self._soft_contact_ke = 1e2 if cfg.soft_contact_ke is None else float(cfg.soft_contact_ke)
        self._soft_contact_kd = 1e-5 if cfg.soft_contact_kd is None else float(cfg.soft_contact_kd)
        self._soft_contact_mu = 0.5 if cfg.soft_contact_mu is None else float(cfg.soft_contact_mu)

        self._rigid_contact_max = (
            100_000 if cfg.rigid_contact_max is None
            else int(cfg.rigid_contact_max)
        )

    # ── Lifecycle helpers ────────────────────────────────────────────

    def _require_setup(self, operation: str) -> _Setup:
        if self._setup is None:
            raise lifecycle_error(
                owner="newton_vbd",
                operation=operation,
                expected="initialize() must run before this operation",
            )
        return self._setup

    def _require_runtime(self, operation: str) -> _Runtime:
        if self._runtime is None:
            raise lifecycle_error(
                owner="newton_vbd",
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
        if self._debug_soft_no_deformation:
            LOGGER.info(
                "[NewtonVBD] DEBUG: soft body deformation gradient DISABLED "
                "(positions only, no cov/rot update)"
            )
        else:
            LOGGER.info(
                f"[NewtonVBD] F singular value clamp: "
                f"[{self._sv_clamp_min}, {self._sv_clamp_max}]"
            )

        builder = newton.ModelBuilder()
        contact_margin = float(self._cfg.contact_margin)
        builder.default_shape_cfg.contact_margin = contact_margin

        LOGGER.info(
            f"[NewtonVBD] collision_geometry={self._collision_geo}, "
            f"contact_margin={contact_margin}"
        )

        self._setup = _Setup(
            builder=builder,
            init_positions=positions.clone().to(self._device),
            init_covariances=covariances.clone().to(self._device),
            init_quats=init_quats.clone().to(self._device) if init_quats is not None else None,
            init_scales=init_scales.clone().to(self._device) if init_scales is not None else None,
            n_particles=int(positions.shape[0]),
        )

    def set_material(self, spec: MaterialSetupSpec) -> None:
        setup = self._require_setup("set_material")
        setup.gravity = tuple(float(x) for x in spec.gravity)

        if not spec.per_part:
            # No per-object definitions → single default rigid body.
            base_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5)
            self._create_rigid_body(
                setup=setup,
                particle_indices=list(range(setup.n_particles)),
                shape_cfg=base_cfg,
                name="single_body",
            )
            return

        for info in spec.per_part:
            material = info.material
            if not isinstance(material, VBDMaterial):
                detail = (
                    f"object={info.name} material_type={type(material).__name__}"
                )
                raise configuration_error(
                    owner="newton_vbd",
                    operation="set_material",
                    expected="newton_vbd requires VBDMaterial per object",
                    detail=detail,
                )
            body = material.body
            if isinstance(body, VBDRigidBody):
                cfg = newton.ModelBuilder.ShapeConfig(
                    density=float(body.density),
                    mu=float(body.mu),
                )
                self._create_rigid_body(
                    setup=setup,
                    particle_indices=list(info.particle_indices),
                    shape_cfg=cfg,
                    name=info.name,
                    collision_geo=body.collision_geometry,
                    kinematic=bool(body.kinematic),
                )
            elif isinstance(body, VBDSoftBody):
                self._create_soft_body(
                    setup=setup,
                    particle_indices=list(info.particle_indices),
                    material=body,
                    name=info.name,
                )
            else:
                detail = f"object={info.name} body_type={type(body).__name__}"
                raise configuration_error(
                    owner="newton_vbd",
                    operation="set_material",
                    expected="VBDMaterial.body must be VBDRigidBody | VBDSoftBody",
                    detail=detail,
                )

    def set_boundary_conditions(
        self,
        bcs: list[BoundaryCondition],
        time: TimeConfig,
    ) -> None:
        setup = self._require_setup("set_boundary_conditions")
        if not isinstance(bcs, list):
            detail = f"bc_params_type={type(bcs).__name__}"
            raise configuration_error(
                owner="newton_vbd",
                operation="set_boundary_conditions",
                expected="bcs must be list",
                detail=detail,
            )

        for bc in bcs:
            if isinstance(bc, SurfaceCollider):
                plane, mu = surface_plane_from_bc(bc)
                plane_cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
                setup.builder.add_shape_plane(plane=plane, cfg=plane_cfg)
                LOGGER.info(
                    "[NewtonVBD] Plane: normal=%s, point=%s, mu=%s",
                    bc.normal,
                    bc.point,
                    mu,
                )

    def finalize(self) -> None:
        setup = self._require_setup("finalize")
        if setup.gravity is None:
            detail = "gravity missing; set_material() was not called or invalid"
            LOGGER.error(
                "[NewtonVBD] backend=newton_vbd operation=finalize detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_vbd",
                operation="finalize",
                expected="set_material() must set gravity before finalize()",
                detail=detail,
            )

        # VBD requires coloring
        setup.builder.color()

        model = setup.builder.finalize(device=self._device)
        model.set_gravity(setup.gravity)
        LOGGER.info("[NewtonVBD] gravity=%s", setup.gravity)

        # Soft contact parameters
        model.soft_contact_ke = self._soft_contact_ke
        model.soft_contact_kd = self._soft_contact_kd
        model.soft_contact_mu = self._soft_contact_mu

        # Limit contact buffer
        model.rigid_contact_max = self._rigid_contact_max

        # Create VBD solver
        solver = SolverVBD(
            model,
            iterations=self._solver_iterations,
            particle_enable_self_contact=self._particle_self_contact,
            particle_self_contact_radius=self._particle_self_contact_radius,
            particle_self_contact_margin=self._particle_self_contact_margin,
        )
        LOGGER.info(
            f"[NewtonVBD] SolverVBD: iterations={self._solver_iterations}, "
            f"self_contact={self._particle_self_contact}"
        )
        if self._particle_self_contact:
            LOGGER.info(
                f"[NewtonVBD]   self_contact_radius="
                f"{self._particle_self_contact_radius}, "
                f"margin={self._particle_self_contact_margin}, "
                f"ke={self._soft_contact_ke}"
            )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        collision_pipeline = newton.CollisionPipeline(model, broad_phase="sap")
        contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

        self._runtime = _Runtime(
            model=model,
            solver=solver,
            state_0=state_0,
            state_1=state_1,
            control=control,
            collision_pipeline=collision_pipeline,
            contacts=contacts,
            rigid_bodies=setup.rigid_bodies,
            soft_bodies=setup.soft_bodies,
            n_particles=setup.n_particles,
            init_quats=setup.init_quats,
            init_scales=setup.init_scales,
        )
        # Release setup-only data (builder, init_positions, init_covariances).
        self._setup = None

    def step(self, dt: float, frame: int) -> None:
        rt = self._require_runtime("step")
        rt.state_0.clear_forces()
        rt.contacts = rt.model.collide(rt.state_0, collision_pipeline=rt.collision_pipeline)
        rt.solver.step(rt.state_0, rt.state_1, rt.control, rt.contacts, dt)
        rt.state_0, rt.state_1 = rt.state_1, rt.state_0

    # ── Constraint application ───────────────────────────────────────

    def apply_constraints(self, constraints: list[ResolvedConstraint]) -> None:
        """Translate scene-graph constraints to VBD model operations.

        - ``PinToWorld``: find the tet vertices that deform a soft GS
          particle (its 4 tet corners, weighted by bary coords).  Set
          those vertices' mass to 0 so VBD freezes them in place.
        - ``PinToBody``: same as PinToWorld (tet vertices mass=0) + record
          a ``PinToBodyHook`` that writes particle_q / qd each substep.
        - ``CollideOnly``: validate the target body is kinematic
          (VBD's contact pipeline handles the rest automatically).
        """
        rt = self._require_runtime("apply_constraints")

        for c in constraints:
            if isinstance(c, ResolvedCollideOnly):
                info = self._rigid_info_by_name(rt, c.part_name)
                if info is None:
                    raise configuration_error(
                        owner="newton_vbd",
                        operation="apply_constraints",
                        expected=(
                            f"CollideOnly part '{c.part_name}' must be a rigid "
                            "VBDMaterial part"
                        ),
                    )
                # CollideOnly semantically = "this body only participates as a
                # collider, not under dynamics".  Force-zero its mass/inertia
                # post-finalize so the user doesn't need to also flip
                # VBDRigidBody.kinematic=True.  (The kinematic flag remains
                # available for bodies that must be kinematic independent of
                # any constraint — e.g. externally driven via body_q.)
                if not info.kinematic:
                    self._zero_body_mass(rt, info.body_idx)
                    info.kinematic = True
                    LOGGER.info(
                        "[NewtonVBD] CollideOnly '%s' → auto-kinematized "
                        "(body_inv_mass=0)", c.part_name,
                    )
                continue

            if isinstance(c, (ResolvedPinToWorld, ResolvedPinToBody)):
                tet_verts = self._gs_indices_to_tet_verts(rt, c.particle_indices)
                self._freeze_tet_vertices(rt, tet_verts)

                if isinstance(c, ResolvedPinToBody):
                    body_info = self._rigid_info_by_name(rt, c.body_part_name)
                    if body_info is None:
                        raise configuration_error(
                            owner="newton_vbd",
                            operation="apply_constraints",
                            expected=(
                                f"PinToBody body '{c.body_part_name}' must be a "
                                "rigid VBDMaterial part"
                            ),
                        )
                    # tet_verts are the physical particles whose world pose
                    # must follow this body.  Their initial world positions
                    # come from the current particle_q (= tet rest pose).
                    particle_q_np = rt.state_0.particle_q.numpy()
                    init_world = particle_q_np[tet_verts].astype(np.float64, copy=False)
                    hook = PinToBodyHook(
                        particle_indices=wp.array(
                            tet_verts.astype(np.int32), dtype=int, device=self._device,
                        ),
                        body_idx=body_info.body_idx,
                        initial_world_positions=init_world,
                    )
                    rt.pin_to_body_hooks.append(hook)
                continue

            raise NotImplementedError(
                f"NewtonVBDBackend: unsupported constraint {type(c).__name__}"
            )

    def pre_step(self, dt: float, frame: int) -> None:
        rt = self._runtime
        if rt is None or not rt.pin_to_body_hooks:
            return
        for hook in rt.pin_to_body_hooks:
            launch_pin_hook(
                hook,
                body_q=rt.state_0.body_q,
                body_qd=rt.state_0.body_qd,
                body_com=rt.model.body_com,
                particle_q=rt.state_0.particle_q,
                particle_qd=rt.state_0.particle_qd,
                device=self._device,
            )

    # ── Constraint helpers ───────────────────────────────────────────

    def _rigid_info_by_name(self, rt: _Runtime, name: str) -> Optional[_RigidInfo]:
        for info in rt.rigid_bodies:
            if info.name == name:
                return info
        return None

    def _gs_indices_to_tet_verts(
        self,
        rt: _Runtime,
        gs_indices: np.ndarray,
    ) -> np.ndarray:
        """Map GS particle indices to the physical tet vertices that drive them.

        Each soft-body GS particle lives inside one tet and is reconstructed
        by barycentric interpolation over the 4 tet vertices.  Pinning the
        GS particle in place means freezing those 4 vertices.

        GS particles that belong to a rigid body have no corresponding
        tet vertex — pinning them to world is meaningless (the whole body
        would need to be kinematic instead); we raise a clear error.
        """
        gs_set = set(int(i) for i in gs_indices)
        tet_vert_ids: set[int] = set()
        consumed: set[int] = set()

        for soft in rt.soft_bodies:
            # Build one dict per soft body: gs_index -> local_pos (O(N) total).
            # Previously we did `soft.particle_indices.index(gi)` per hit,
            # which is O(N*M) and blew up on alocasia-scale clouds.
            gi_to_local = {int(gi): lp for lp, gi in enumerate(soft.particle_indices)}
            hit = [gi for gi in gs_set if gi in gi_to_local]
            if not hit:
                continue
            for gi in hit:
                lp = gi_to_local[gi]
                tet_id = int(soft.tet_ids[lp])
                for v in soft.tet_cells[tet_id]:
                    tet_vert_ids.add(int(v) + soft.vert_offset)
            consumed.update(hit)

        missed = gs_set - consumed
        if missed:
            # Identify which rigid bodies the missed GS live in (if any)
            for info in rt.rigid_bodies:
                rigid_gs = set(int(i) for i in info.particle_indices)
                stuck = missed & rigid_gs
                if stuck:
                    raise configuration_error(
                        owner="newton_vbd",
                        operation="apply_constraints",
                        expected=(
                            "cannot pin individual particles of a rigid body "
                            f"'{info.name}'; pin the whole body via "
                            "CollideOnly+kinematic, or select only soft-body GS"
                        ),
                    )
                missed -= rigid_gs
            if missed:
                raise configuration_error(
                    owner="newton_vbd",
                    operation="apply_constraints",
                    expected=(
                        f"{len(missed)} selected GS indices belong to no "
                        "known soft/rigid body"
                    ),
                )

        if not tet_vert_ids:
            raise configuration_error(
                owner="newton_vbd",
                operation="apply_constraints",
                expected="constraint resolved to zero physical tet vertices",
            )
        return np.array(sorted(tet_vert_ids), dtype=np.int64)

    def _freeze_tet_vertices(self, rt: _Runtime, vert_indices: np.ndarray) -> None:
        """Set ``particle_mass`` / ``particle_inv_mass`` to 0 for given verts."""
        if vert_indices.size == 0:
            return
        mass_np = rt.model.particle_mass.numpy()
        inv_np = rt.model.particle_inv_mass.numpy()
        mass_np[vert_indices] = 0.0
        inv_np[vert_indices] = 0.0
        rt.model.particle_mass = wp.array(
            mass_np, dtype=float, device=self._device,
        )
        rt.model.particle_inv_mass = wp.array(
            inv_np, dtype=float, device=self._device,
        )
        LOGGER.info(
            "[NewtonVBD] Froze %d tet vertices (mass=0)", int(vert_indices.size),
        )

    def _zero_body_mass(self, rt: _Runtime, body_idx: int) -> None:
        """Post-finalize: zero mass/inertia of a body to make it kinematic.

        Used by CollideOnly's apply_constraints branch so users don't need
        to also flip VBDRigidBody.kinematic=True at config time.
        """
        for name in ("body_mass", "body_inv_mass"):
            arr = getattr(rt.model, name)
            a_np = arr.numpy()
            a_np[body_idx] = 0.0
            setattr(rt.model, name, wp.array(a_np, dtype=float, device=self._device))
        for name in ("body_inertia", "body_inv_inertia"):
            arr = getattr(rt.model, name)
            a_np = arr.numpy()
            a_np[body_idx] = np.zeros((3, 3), dtype=a_np.dtype)
            setattr(rt.model, name, wp.array(a_np, dtype=wp.mat33, device=self._device))

    def get_state(self) -> SimulationState:
        rt = self._require_runtime("get_state")
        state, rt.frame_counter = export_state(
            state_0=rt.state_0,
            rigid_bodies=rt.rigid_bodies,
            soft_bodies=rt.soft_bodies,
            n_particles=rt.n_particles,
            device=self._device,
            init_quats=rt.init_quats,
            init_scales=rt.init_scales,
            frame_counter=rt.frame_counter,
            debug_soft_no_deformation=self._debug_soft_no_deformation,
            sv_clamp_min=self._sv_clamp_min,
            sv_clamp_max=self._sv_clamp_max,
        )
        return state

    # ── Rigid body creation ──────────────────────────────────────────

    def _create_rigid_body(
        self,
        *,
        setup: _Setup,
        particle_indices: list[int],
        shape_cfg: newton.ModelBuilder.ShapeConfig,
        name: str,
        collision_geo: str | None = None,
        kinematic: bool = False,
    ) -> None:
        result = create_rigid_body(
            builder=setup.builder,
            init_positions=setup.init_positions,
            init_covariances=setup.init_covariances,
            init_quats=setup.init_quats,
            init_scales=setup.init_scales,
            device=self._device,
            particle_indices=particle_indices,
            shape_cfg=shape_cfg,
            name=name,
            collision_geo=collision_geo,
            default_collision_geo=self._collision_geo,
            alpha=self._alpha,
            max_triangles=self._max_triangles,
        )
        if kinematic:
            # Zero-mass body: treated as kinematic by Newton solvers.
            # body_mass / body_inv_mass are per-body arrays on the builder.
            setup.builder.body_mass[result.body_idx] = 0.0
            setup.builder.body_inv_mass[result.body_idx] = 0.0
            setup.builder.body_inertia[result.body_idx] = np.zeros((3, 3), dtype=np.float32)
            setup.builder.body_inv_inertia[result.body_idx] = np.zeros((3, 3), dtype=np.float32)
            LOGGER.info(
                "[NewtonVBD] Rigid '%s' set kinematic (body_inv_mass=0)", name,
            )
        setup.rigid_bodies.append(
            _RigidInfo(
                body_idx=result.body_idx,
                particle_indices=particle_indices,
                init_local_pos=result.init_local_pos,
                init_cov_3x3=result.init_cov_3x3,
                init_quats=result.init_quats,
                init_scales=result.init_scales,
                name=name,
                kinematic=kinematic,
            )
        )

    # ── Soft body creation ───────────────────────────────────────────

    def _create_soft_body(
        self,
        *,
        setup: _Setup,
        particle_indices: list[int],
        material: VBDSoftBody,
        name: str,
    ) -> None:
        """Create a soft body using Newton's ``add_soft_grid``.

        Generates a uniform regular tet grid covering the GS particle
        bounding box and embeds GS particles via barycentric coordinates.
        """
        if len(particle_indices) == 0:
            detail = f"object={name} particle_count=0"
            LOGGER.error(
                "[NewtonVBD] backend=newton_vbd operation=_create_soft_body "
                "detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_vbd",
                operation="_create_soft_body",
                expected="soft object must contain at least one particle",
                detail=detail,
            )

        idx_t = torch.tensor(particle_indices, dtype=torch.long)
        pos = setup.init_positions[idx_t].float()
        pos_np = pos.detach().cpu().numpy()
        try:
            embedding = build_soft_grid_embedding(pos_np=pos_np, material=material)
        except (TypeError, ValueError) as exc:
            detail = f"object={name} reason={exc}"
            LOGGER.error(
                "[NewtonVBD] backend=newton_vbd operation=_create_soft_body "
                "detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_vbd",
                operation="_create_soft_body",
                expected="valid soft grid/material parameters",
                detail=detail,
            ) from exc

        vert_offset = setup.particle_offset

        setup.builder.add_soft_grid(
            pos=wp.vec3(
                float(embedding.spec.bbox_min[0]),
                float(embedding.spec.bbox_min[1]),
                float(embedding.spec.bbox_min[2]),
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=embedding.spec.dim_x,
            dim_y=embedding.spec.dim_y,
            dim_z=embedding.spec.dim_z,
            cell_x=embedding.spec.cell_x,
            cell_y=embedding.spec.cell_y,
            cell_z=embedding.spec.cell_z,
            density=embedding.spec.density,
            k_mu=embedding.spec.k_mu,
            k_lambda=embedding.spec.k_lambda,
            k_damp=embedding.spec.k_damp,
        )

        vert_count = embedding.vert_count
        setup.particle_offset += vert_count

        LOGGER.info(
            f"[NewtonVBD] Embedding {len(particle_indices)} GS particles "
            f"in {embedding.tet_cells.shape[0]} tets..."
        )

        if embedding.outside_count > 0:
            pct = 100.0 * embedding.outside_count / len(particle_indices)
            LOGGER.warning(
                f"[NewtonVBD] WARNING: {embedding.outside_count}/{len(particle_indices)} "
                f"({pct:.1f}%) GS particles outside grid → clamped"
            )
        else:
            LOGGER.info(
                f"[NewtonVBD] All {len(particle_indices)} particles "
                f"inside grid"
            )

        cov6 = setup.init_covariances[idx_t]

        setup.soft_bodies.append(_SoftInfo(
            particle_indices=particle_indices,
            tet_ids=embedding.tet_ids,
            bary_coords=embedding.bary_coords,
            vert_offset=vert_offset,
            vert_count=vert_count,
            tet_cells=embedding.tet_cells,
            rest_verts=embedding.grid_vertices.copy(),
            init_cov_6=cov6,
            name=name,
        ))
        LOGGER.info(
            f"[NewtonVBD] Soft '{name}': {len(particle_indices)} GS particles, "
            f"grid {embedding.spec.dim_x}x{embedding.spec.dim_y}x{embedding.spec.dim_z} = "
            f"{vert_count} verts, {embedding.tet_cells.shape[0]} tets, "
            f"cell=[{embedding.spec.cell_x:.4f},{embedding.spec.cell_y:.4f},"
            f"{embedding.spec.cell_z:.4f}], "
            f"density={embedding.spec.density}, k_mu={embedding.spec.k_mu:.0e}, "
            f"k_lambda={embedding.spec.k_lambda:.0e}, k_damp={embedding.spec.k_damp:.0e}"
        )
