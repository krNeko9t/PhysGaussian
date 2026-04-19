"""
Newton implicit-MPM backend adapter that implements the PhysicsBackend interface.

This wraps Newton's SolverImplicitMPM and exposes a clean, backend-agnostic API
through PhysicsBackend / SimulationState, matching the existing WarpMPMBackend.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import torch

import newton
from newton.solvers import SolverImplicitMPM

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.newton_mpm.boundary_conditions import (
    BoundaryConditionRuntime,
    register_boundary_conditions,
)
from physics_sim.backend.newton_mpm.initialization import build_mpm_initialization
from physics_sim.backend.newton_mpm.materials import apply_material_to_model, apply_solver_options
from physics_sim.backend.newton_mpm.state_export import export_mpm_state
from physics_sim.errors import configuration_error, lifecycle_error
from physics_sim.logging_utils import get_logger

LOGGER = get_logger(__name__)


class NewtonMPMBackend(PhysicsBackend):
    """Physics backend powered by Newton's implicit MPM solver."""

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._builder: newton.ModelBuilder | None = None
        self._model: newton.Model | None = None
        self._solver: SolverImplicitMPM | None = None
        self._state_0: newton.State | None = None
        self._state_1: newton.State | None = None
        self._control: newton.Control | None = None

        # Covariance tracking (Newton doesn't store per-particle cov natively)
        self._init_cov: wp.array | None = None   # (N*6,) float, flat
        self._out_cov: wp.array | None = None     # (N*6,) float, flat
        self._out_R: wp.array | None = None       # (N,) mat33

        # 2DGS support: quats + scales output from SVD
        self._init_quats_wp: wp.array | None = None    # (N,) vec4 wxyz
        self._init_scales_wp: wp.array | None = None   # (N*S,) float flat
        self._out_quats_wp: wp.array | None = None     # (N,) vec4 wxyz
        self._out_scales_wp: wp.array | None = None    # (N*S,) float flat
        self._num_scales: int = 0

        self._n_particles: int = 0
        self._grid_lim: float = 2.0
        self._time: float = 0.0
        self._substep_dt: float = 0.0  # last substep dt, for deferred frame update
        self._frames_dirty: bool = False  # whether particle frames need updating

        # Solver options — start from Newton's own defaults.
        # Overrides come from config JSON ("newton_mpm" → "solver" section)
        # and are applied in set_material().
        self._solver_opts = SolverImplicitMPM.Config()
        self._material_params: dict | None = None
        self._cov_np: np.ndarray | None = None
        self._volumes: np.ndarray | None = None
        self._bc_params: list = []
        self._time_params: dict = {}
        self._bc_runtime = BoundaryConditionRuntime()

    # ------------------------------------------------------------------
    # PhysicsBackend interface
    # ------------------------------------------------------------------

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
        """Build a Newton Model from the preprocessed GS particle data.

        Args:
            positions:   (N, 3) particle positions in rotated world space.
            volumes:     (N,)   per-particle volumes (used for mass).
            covariances: (N, 6) upper-triangle covariance matrices.
            n_grid:      Grid resolution (default 100).
            grid_lim:    Ignored (kept for interface compat).  voxel_size
                         is computed from particle bounding box / n_grid.
        """
        self._n_particles = int(positions.shape[0])
        init_result = build_mpm_initialization(
            positions=positions,
            volumes=volumes,
            covariances=covariances,
            n_grid=n_grid,
        )
        self._grid_lim = init_result.grid_lim
        self._bbox_lo = init_result.bbox_lo
        self._bbox_hi = init_result.bbox_hi
        self._solver_opts.voxel_size = init_result.voxel_size
        LOGGER.info(
            "[NewtonMPM] bbox_extent=%.4f voxel_size=%.6f",
            init_result.grid_lim,
            init_result.voxel_size,
        )

        # Store builder + arrays; finalize() will create model + solver.
        self._builder = init_result.builder
        self._cov_np = init_result.cov_flat
        self._volumes = init_result.volumes

        # 2DGS: store initial quats + scales for SVD-based output
        iq = kwargs.get("init_quats")
        isc = kwargs.get("init_scales")
        if (iq is None) ^ (isc is None):
            detail = (
                f"init_quats_provided={iq is not None} "
                f"init_scales_provided={isc is not None}"
            )
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=initialize detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_mpm",
                operation="initialize",
                expected="init_quats and init_scales must be provided together",
                detail=detail,
            )
        if iq is not None and isc is not None:
            self._init_quats_np = iq.detach().cpu().numpy().astype(np.float32)
            self._init_scales_np = isc.detach().cpu().numpy().astype(np.float32)
            self._num_scales = isc.shape[1]
        else:
            self._init_quats_np = None
            self._init_scales_np = None

    def set_material(self, material_params: dict) -> None:
        """Configure Newton MPM material from the existing config dict.

        Supports the same keys as WarpMPMBackend:
            material, E, nu, density, friction_angle, yield_stress,
            hardening, g, rpic_damping, grid_v_damping_scale, etc.
        """
        if not isinstance(material_params, dict):
            detail = f"material_params_type={type(material_params).__name__}"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=set_material detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_mpm",
                operation="set_material",
                expected="material_params must be dict",
                detail=detail,
            )

        # Store material params; applied to the finalized model in finalize().
        self._material_params = dict(material_params)
        apply_solver_options(
            solver_opts=self._solver_opts,
            material_params=self._material_params,
        )

    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
        """Register BCs and build per-step BC runtime."""
        if not isinstance(bc_params, list):
            detail = f"bc_params_type={type(bc_params).__name__}"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=set_boundary_conditions "
                "detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_mpm",
                operation="set_boundary_conditions",
                expected="bc_params must be list",
                detail=detail,
            )
        if not isinstance(time_params, dict):
            detail = f"time_params_type={type(time_params).__name__}"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=set_boundary_conditions "
                "detail=%s",
                detail,
            )
            raise configuration_error(
                owner="newton_mpm",
                operation="set_boundary_conditions",
                expected="time_params must be dict",
                detail=detail,
            )
        for bc in bc_params:
            if not isinstance(bc, dict):
                detail = f"bc_item_type={type(bc).__name__}"
                LOGGER.error(
                    "[NewtonMPM] backend=newton_mpm operation=set_boundary_conditions "
                    "detail=%s",
                    detail,
                )
                raise configuration_error(
                    owner="newton_mpm",
                    operation="set_boundary_conditions",
                    expected="each bc item must be dict",
                    detail=detail,
                )

        self._bc_params = bc_params
        self._time_params = time_params

        builder = self._builder
        if builder is None:
            detail = "builder missing; initialize() not completed"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=set_boundary_conditions "
                "detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_mpm",
                operation="set_boundary_conditions",
                expected="initialize() must run first",
                detail=detail,
            )
        self._bc_runtime = register_boundary_conditions(
            builder=builder,
            bc_params=bc_params,
            voxel_size=float(self._solver_opts.voxel_size),
            bbox_lo=self._bbox_lo,
            bbox_hi=self._bbox_hi,
        )

    def finalize(self) -> None:
        """Create the Newton solver and initial states. Must be called after
        set_material() and set_boundary_conditions()."""
        builder = self._builder
        if builder is None:
            detail = "builder missing; initialize() not completed"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=finalize detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="initialize() must run first",
                detail=detail,
            )
        if self._material_params is None:
            detail = "material params missing; set_material() not called"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=finalize detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="set_material() must run first",
                detail=detail,
            )

        # Finalize model after all shapes are registered.
        self._model = builder.finalize(device=self._device)
        self._builder = None
        model = self._model

        # ── Covariance tracking buffers ───────────────────────────────
        n = self._n_particles
        if self._cov_np is None:
            detail = "covariance buffer missing from initialize()"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=finalize detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="initialize() must provide covariances",
                detail=detail,
            )
        self._init_cov = wp.from_numpy(self._cov_np, dtype=float, device=self._device)
        self._out_cov = wp.zeros(n * 6, dtype=float, device=self._device)
        self._out_R = wp.zeros(n, dtype=wp.mat33, device=self._device)

        # ── 2DGS quats/scales buffers ─────────────────────────────────
        if self._init_quats_np is not None:
            self._init_quats_wp = wp.from_numpy(
                self._init_quats_np.reshape(-1, 4), dtype=wp.vec4, device=self._device,
            )
            self._init_scales_wp = wp.from_numpy(
                self._init_scales_np.reshape(-1), dtype=float, device=self._device,
            )
            self._out_quats_wp = wp.zeros(n, dtype=wp.vec4, device=self._device)
            self._out_scales_wp = wp.zeros(
                n * self._num_scales, dtype=float, device=self._device,
            )

        # Apply material params now that model exists.
        if self._volumes is None or self._material_params is None:
            detail = "volumes/material incomplete before apply_material_to_model"
            LOGGER.error(
                "[NewtonMPM] backend=newton_mpm operation=finalize detail=%s",
                detail,
            )
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="initialize()+set_material() must provide volumes/material",
                detail=detail,
            )
        apply_material_to_model(
            model=self._model,
            volumes=self._volumes,
            material_params=self._material_params,
            device=self._device,
        )

        # ── Create solver ────────────────────────────────────────────
        self._solver = SolverImplicitMPM(model, self._solver_opts)

        # ── Create double-buffered states + control ──────────────────
        self._state_0 = model.state()
        self._state_1 = model.state()
        self._control = model.control()
        self._time = 0.0

        self._bc_runtime.log_unimplemented()

    def step(self, dt: float, frame: int) -> None:
        """Advance simulation by one substep."""
        self._bc_runtime.apply_velocity_bcs(
            state=self._state_0,
            time_now=self._time,
            dt=dt,
            device=self._device,
        )
        self._bc_runtime.apply_impulse_bcs(
            state=self._state_0,
            particle_mass=self._model.particle_mass,
            time_now=self._time,
            dt=dt,
            device=self._device,
        )

        # Newton solver step
        self._solver.step(self._state_0, self._state_1, self._control, None, dt)

        # Swap states (defer update_particle_frames to get_state for performance)
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._substep_dt = dt
        self._frames_dirty = True
        self._time += dt

    def get_state(self) -> SimulationState:
        state, self._frames_dirty = export_mpm_state(
            state=self._state_0,
            solver=self._solver,
            prev_state=self._state_1,
            substep_dt=self._substep_dt,
            frames_dirty=self._frames_dirty,
            n_particles=self._n_particles,
            init_cov=self._init_cov,
            out_cov=self._out_cov,
            out_R=self._out_R,
            device=self._device,
            init_quats_wp=self._init_quats_wp,
            init_scales_wp=self._init_scales_wp,
            out_quats_wp=self._out_quats_wp,
            out_scales_wp=self._out_scales_wp,
            num_scales=self._num_scales,
        )
        return state

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> newton.Model:
        return self._model

    @property
    def raw_solver(self) -> SolverImplicitMPM:
        return self._solver
