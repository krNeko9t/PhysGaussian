"""
Newton implicit-MPM backend adapter that implements the PhysicsBackend interface.

This wraps Newton's SolverImplicitMPM and exposes a clean, backend-agnostic API
through PhysicsBackend / SimulationState.

Phase C decomposition (option α): the backend holds a mutable ``_Setup``
during configuration and, after ``finalize()``, a ``_Runtime`` containing
only the fields needed by ``step`` / ``get_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import warp as wp
import torch

import newton
from newton.solvers import SolverImplicitMPM

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.newton_common.pin_kernel import (
    PinToBodyHook,
    launch_pin_hook,
)
from physics_sim.backend.newton_mpm.boundary_conditions import (
    BoundaryConditionRuntime,
    register_boundary_conditions,
)
from physics_sim.backend.newton_mpm.initialization import build_mpm_initialization
from physics_sim.backend.newton_mpm.materials import apply_material_to_model
from physics_sim.backend.newton_mpm.state_export import export_mpm_state
from physics_sim.backend.spec import MaterialSetupSpec
from physics_sim.config.models import (
    BoundaryCondition,
    NewtonMPMConfig,
    TimeConfig,
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


@dataclass
class _Setup:
    """Mutable configuration-phase state (``initialize`` → ``finalize``)."""

    builder: newton.ModelBuilder
    cov_np: np.ndarray
    volumes: np.ndarray
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray
    n_particles: int
    init_quats_np: Optional[np.ndarray] = None
    init_scales_np: Optional[np.ndarray] = None
    num_scales: int = 0
    setup_spec: Optional[MaterialSetupSpec] = None
    bc_runtime: BoundaryConditionRuntime = field(default_factory=BoundaryConditionRuntime)


@dataclass
class _Runtime:
    """Runtime-phase state produced by ``finalize``."""

    model: "newton.Model"
    solver: SolverImplicitMPM
    state_0: "newton.State"
    state_1: "newton.State"
    control: "newton.Control"
    n_particles: int
    init_cov: "wp.array"
    out_cov: "wp.array"
    out_R: "wp.array"
    init_quats_wp: Optional["wp.array"]
    init_scales_wp: Optional["wp.array"]
    out_quats_wp: Optional["wp.array"]
    out_scales_wp: Optional["wp.array"]
    num_scales: int
    bc_runtime: BoundaryConditionRuntime
    time: float = 0.0
    substep_dt: float = 0.0
    frames_dirty: bool = False
    pin_to_body_hooks: list[PinToBodyHook] = field(default_factory=list)


class NewtonMPMBackend(PhysicsBackend):
    """Physics backend powered by Newton's implicit MPM solver."""

    def __init__(self, *, cfg: NewtonMPMConfig, device: str = "cuda:0"):
        self._cfg = cfg
        self._device = device
        self._setup: Optional[_Setup] = None
        self._runtime: Optional[_Runtime] = None

        # Solver options — start from Newton's own defaults + cfg overrides.
        self._solver_opts = SolverImplicitMPM.Config()
        if cfg.solver_iterations is not None:
            try:
                self._solver_opts.max_iterations = int(cfg.solver_iterations)
            except AttributeError:
                pass
        if cfg.solver_tolerance is not None:
            try:
                self._solver_opts.tolerance = float(cfg.solver_tolerance)
            except AttributeError:
                pass
        if cfg.transfer_scheme is not None:
            self._solver_opts.transfer_scheme = str(cfg.transfer_scheme)

    # ── Lifecycle helpers ────────────────────────────────────────────

    def _require_setup(self, operation: str) -> _Setup:
        if self._setup is None:
            raise lifecycle_error(
                owner="newton_mpm",
                operation=operation,
                expected="initialize() must run before this operation",
            )
        return self._setup

    def _require_runtime(self, operation: str) -> _Runtime:
        if self._runtime is None:
            raise lifecycle_error(
                owner="newton_mpm",
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
        init_quats: torch.Tensor | None = None,
        init_scales: torch.Tensor | None = None,
    ) -> None:
        """Build a Newton Model from the preprocessed GS particle data."""
        init_result = build_mpm_initialization(
            positions=positions,
            volumes=volumes,
            covariances=covariances,
            n_grid=int(self._cfg.n_grid),
        )
        self._solver_opts.voxel_size = init_result.voxel_size
        LOGGER.info(
            "[NewtonMPM] bbox_extent=%.4f voxel_size=%.6f",
            init_result.grid_lim,
            init_result.voxel_size,
        )

        # 2DGS: store initial quats + scales for SVD-based output
        if (init_quats is None) ^ (init_scales is None):
            detail = (
                f"init_quats_provided={init_quats is not None} "
                f"init_scales_provided={init_scales is not None}"
            )
            raise configuration_error(
                owner="newton_mpm",
                operation="initialize",
                expected="init_quats and init_scales must be provided together",
                detail=detail,
            )
        init_quats_np = None
        init_scales_np = None
        num_scales = 0
        if init_quats is not None and init_scales is not None:
            init_quats_np = init_quats.detach().cpu().numpy().astype(np.float32)
            init_scales_np = init_scales.detach().cpu().numpy().astype(np.float32)
            num_scales = int(init_scales.shape[1])

        self._setup = _Setup(
            builder=init_result.builder,
            cov_np=init_result.cov_flat,
            volumes=init_result.volumes,
            bbox_lo=init_result.bbox_lo,
            bbox_hi=init_result.bbox_hi,
            n_particles=int(positions.shape[0]),
            init_quats_np=init_quats_np,
            init_scales_np=init_scales_np,
            num_scales=num_scales,
        )

    def set_material(self, spec: MaterialSetupSpec) -> None:
        """Store material setup spec; applied to the finalized model in finalize()."""
        setup = self._require_setup("set_material")
        setup.setup_spec = spec

    def set_boundary_conditions(
        self,
        bcs: list[BoundaryCondition],
        time: TimeConfig,
    ) -> None:
        """Register BCs and build per-step BC runtime."""
        setup = self._require_setup("set_boundary_conditions")
        if not isinstance(bcs, list):
            detail = f"bc_params_type={type(bcs).__name__}"
            raise configuration_error(
                owner="newton_mpm",
                operation="set_boundary_conditions",
                expected="bcs must be list",
                detail=detail,
            )

        setup.bc_runtime = register_boundary_conditions(
            builder=setup.builder,
            bc_params=bcs,
            voxel_size=float(self._solver_opts.voxel_size),
            bbox_lo=setup.bbox_lo,
            bbox_hi=setup.bbox_hi,
        )

    def finalize(self) -> None:
        """Drain ``_Setup`` into ``_Runtime`` (model/solver/states)."""
        setup = self._require_setup("finalize")
        if setup.setup_spec is None:
            detail = "material setup spec missing; set_material() not called"
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="set_material() must run first",
                detail=detail,
            )

        model = setup.builder.finalize(device=self._device)

        # ── Covariance tracking buffers ───────────────────────────────
        n = setup.n_particles
        init_cov = wp.from_numpy(setup.cov_np, dtype=float, device=self._device)
        out_cov = wp.zeros(n * 6, dtype=float, device=self._device)
        out_R = wp.zeros(n, dtype=wp.mat33, device=self._device)

        # ── 2DGS quats/scales buffers ─────────────────────────────────
        init_quats_wp = None
        init_scales_wp = None
        out_quats_wp = None
        out_scales_wp = None
        if setup.init_quats_np is not None and setup.init_scales_np is not None:
            init_quats_wp = wp.from_numpy(
                setup.init_quats_np.reshape(-1, 4), dtype=wp.vec4, device=self._device,
            )
            init_scales_wp = wp.from_numpy(
                setup.init_scales_np.reshape(-1), dtype=float, device=self._device,
            )
            out_quats_wp = wp.zeros(n, dtype=wp.vec4, device=self._device)
            out_scales_wp = wp.zeros(
                n * setup.num_scales, dtype=float, device=self._device,
            )

        apply_material_to_model(
            model=model,
            volumes=setup.volumes,
            spec=setup.setup_spec,
            device=self._device,
        )

        solver = SolverImplicitMPM(model, self._solver_opts)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        bc_runtime = setup.bc_runtime
        bc_runtime.log_unimplemented()

        self._runtime = _Runtime(
            model=model,
            solver=solver,
            state_0=state_0,
            state_1=state_1,
            control=control,
            n_particles=n,
            init_cov=init_cov,
            out_cov=out_cov,
            out_R=out_R,
            init_quats_wp=init_quats_wp,
            init_scales_wp=init_scales_wp,
            out_quats_wp=out_quats_wp,
            out_scales_wp=out_scales_wp,
            num_scales=setup.num_scales,
            bc_runtime=bc_runtime,
        )
        # Release setup-only data (builder, cov_np, volumes, bbox).
        self._setup = None

    def step(self, dt: float, frame: int) -> None:
        """Advance simulation by one substep."""
        rt = self._require_runtime("step")
        rt.bc_runtime.apply_velocity_bcs(
            state=rt.state_0,
            time_now=rt.time,
            dt=dt,
            device=self._device,
        )
        rt.bc_runtime.apply_impulse_bcs(
            state=rt.state_0,
            particle_mass=rt.model.particle_mass,
            time_now=rt.time,
            dt=dt,
            device=self._device,
        )

        rt.solver.step(rt.state_0, rt.state_1, rt.control, None, dt)
        # Swap states (defer update_particle_frames to get_state for performance)
        rt.state_0, rt.state_1 = rt.state_1, rt.state_0
        rt.substep_dt = dt
        rt.frames_dirty = True
        rt.time += dt

    def get_state(self) -> SimulationState:
        rt = self._require_runtime("get_state")
        state, rt.frames_dirty = export_mpm_state(
            state=rt.state_0,
            solver=rt.solver,
            prev_state=rt.state_1,
            substep_dt=rt.substep_dt,
            frames_dirty=rt.frames_dirty,
            n_particles=rt.n_particles,
            init_cov=rt.init_cov,
            out_cov=rt.out_cov,
            out_R=rt.out_R,
            device=self._device,
            init_quats_wp=rt.init_quats_wp,
            init_scales_wp=rt.init_scales_wp,
            out_quats_wp=rt.out_quats_wp,
            out_scales_wp=rt.out_scales_wp,
            num_scales=rt.num_scales,
        )
        return state

    # ── Constraint application ───────────────────────────────────────

    def apply_constraints(self, constraints: list[ResolvedConstraint]) -> None:
        """Translate scene-graph constraints to MPM model operations.

        MPM particles are 1:1 with GS particles, so ``PinToWorld`` and
        ``PinToBody`` directly operate on ``model.particle_mass`` at the
        GS indices produced by the resolver.

        ``CollideOnly`` currently raises NotImplementedError: the MPM
        backend has no mesh-collider path yet (see Phase F2 task).
        """
        rt = self._require_runtime("apply_constraints")

        for c in constraints:
            if isinstance(c, ResolvedCollideOnly):
                raise NotImplementedError(
                    "NewtonMPMBackend does not yet support CollideOnly "
                    f"(part='{c.part_name}'). Track Phase F2."
                )

            if isinstance(c, (ResolvedPinToWorld, ResolvedPinToBody)):
                self._freeze_particles(rt, c.particle_indices)

                if isinstance(c, ResolvedPinToBody):
                    # MPM has no rigid bodies in the current builder path,
                    # so there is no body_q to follow.  When Phase F2
                    # adds rigid bodies, this branch becomes analogous
                    # to the VBD version.
                    raise NotImplementedError(
                        "NewtonMPMBackend does not yet expose rigid bodies, "
                        f"so PinToBody (body='{c.body_part_name}') is "
                        "unsupported. Track Phase F2."
                    )
                continue

            raise NotImplementedError(
                f"NewtonMPMBackend: unsupported constraint {type(c).__name__}"
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

    def _freeze_particles(self, rt: _Runtime, indices: np.ndarray) -> None:
        """Set ``particle_mass`` / ``particle_inv_mass`` to 0 for given indices.

        MPM's kernels treat particles with mass=0 as kinematic (infinite
        inverse mass → velocity zeroed during integration, strains frozen
        during rasterization).  This is the sole mechanism for pinning.
        """
        if indices.size == 0:
            return
        mass_np = rt.model.particle_mass.numpy()
        inv_np = rt.model.particle_inv_mass.numpy()
        mass_np[indices] = 0.0
        inv_np[indices] = 0.0
        rt.model.particle_mass = wp.array(
            mass_np, dtype=float, device=self._device,
        )
        rt.model.particle_inv_mass = wp.array(
            inv_np, dtype=float, device=self._device,
        )
        LOGGER.info(
            "[NewtonMPM] Froze %d particles (mass=0)", int(indices.size),
        )

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> newton.Model:
        rt = self._require_runtime("raw_model")
        return rt.model

    @property
    def raw_solver(self) -> SolverImplicitMPM:
        rt = self._require_runtime("raw_solver")
        return rt.solver
