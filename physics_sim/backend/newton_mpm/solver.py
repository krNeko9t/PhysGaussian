"""
Newton implicit-MPM backend adapter that implements the PhysicsBackend interface.

This wraps Newton's SolverImplicitMPM and exposes a clean, backend-agnostic API
through PhysicsBackend / SimulationState, matching the existing WarpMPMBackend.
"""

from __future__ import annotations

import math
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
from physics_sim.backend.newton_mpm.materials import apply_material_to_model
from physics_sim.backend.newton_mpm.state_export import export_mpm_state
from physics_sim.errors import lifecycle_error
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
        self._n_particles = positions.shape[0]
        n = self._n_particles

        # ── Compute voxel_size from particle bounding box ────────────
        pos_np = positions.detach().cpu().numpy().astype(np.float32)
        lo = pos_np.min(axis=0)
        hi = pos_np.max(axis=0)
        max_extent = float((hi - lo).max())
        if max_extent < 1e-8:
            max_extent = 1.0
        voxel_size = max_extent / max(n_grid, 1)
        self._grid_lim = max_extent
        self._bbox_lo = lo.tolist()
        self._bbox_hi = hi.tolist()
        self._solver_opts.voxel_size = voxel_size
        LOGGER.info(
            "[NewtonMPM] bbox_extent=%.4f voxel_size=%.6f", max_extent, voxel_size
        )

        # ── Build Newton ModelBuilder (finalized in finalize()) ──────
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        vol_np = volumes.detach().cpu().numpy().astype(np.float32)

        # Compute per-particle mass from volume (density set later in set_material)
        # Use a placeholder density of 1.0; actual mass will be recomputed.
        mass_np = vol_np.copy()  # mass = vol * density, density=1 placeholder

        # Compute radius from volume (sphere approximation)
        radius_np = np.cbrt(vol_np * 3.0 / (4.0 * math.pi)).astype(np.float32)
        radius_np = np.maximum(radius_np, 1e-6)

        # Batch add particles
        pos_list = [wp.vec3(float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2]))
                    for i in range(n)]
        vel_list = [wp.vec3(0.0, 0.0, 0.0)] * n
        mass_list = [float(mass_np[i]) for i in range(n)]
        radius_list = [float(radius_np[i]) for i in range(n)]
        flags_list = [int(newton.ParticleFlags.ACTIVE)] * n

        builder.add_particles(
            pos=pos_list,
            vel=vel_list,
            mass=mass_list,
            radius=radius_list,
            flags=flags_list,
        )

        # Store builder + arrays; finalize() will create model + solver.
        self._builder = builder
        self._cov_np = (
            covariances.detach().cpu().numpy().astype(np.float32).reshape(-1)
        )
        self._volumes = vol_np

        # 2DGS: store initial quats + scales for SVD-based output
        iq = kwargs.get("init_quats")
        isc = kwargs.get("init_scales")
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
        # Store material params; applied to the finalized model in finalize().
        self._material_params = dict(material_params)

        # ── Solver options from material params ──────────────────────
        # Transfer scheme: map rpic_damping → "pic" / "apic"
        rpic = material_params.get("rpic_damping", 0.0)
        if rpic < 0:
            self._solver_opts.transfer_scheme = "pic"
        else:
            self._solver_opts.transfer_scheme = "apic"

        # Apply solver option overrides from config ("newton_mpm" → "solver").
        # Any key that exists as an attribute on SolverImplicitMPM.Config
        # can be overridden here.  Unrecognised keys are warned about.
        newton_opts = material_params.get("newton_solver_opts", {})
        _SOLVER_OPT_TYPES = {
            "max_iterations": int,
            "tolerance": float,
            "solver": str,
            "grid_type": str,
            "transfer_scheme": str,
            "air_drag": float,
            "grid_padding": int,
        }
        for key, val in newton_opts.items():
            if hasattr(self._solver_opts, key):
                cast = _SOLVER_OPT_TYPES.get(key, type(val))
                setattr(self._solver_opts, key, cast(val))
                LOGGER.info("[NewtonMPM] solver.%s=%s", key, cast(val))
            else:
                LOGGER.warning(
                    "[NewtonMPM] unknown solver option '%s', ignored", key
                )

        # NOTE: Material fields on the Newton model are not available until
        # builder.finalize() is called. We apply physical parameters in finalize().

    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
        """Register BCs and build per-step BC runtime."""
        self._bc_params = bc_params
        self._time_params = time_params

        builder = self._builder
        if builder is None:
            raise lifecycle_error(
                owner="newton_mpm",
                operation="set_boundary_conditions",
                expected="initialize() must run first",
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
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="initialize() must run first",
            )
        if self._material_params is None:
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="set_material() must run first",
            )

        # Finalize model after all shapes are registered.
        self._model = builder.finalize(device=self._device)
        self._builder = None
        model = self._model

        # ── Covariance tracking buffers ───────────────────────────────
        n = self._n_particles
        if self._cov_np is None:
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="initialize() must provide covariances",
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
            raise lifecycle_error(
                owner="newton_mpm",
                operation="finalize",
                expected="initialize()+set_material() must provide volumes/material",
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
