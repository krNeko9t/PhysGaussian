"""
Warp-MPM backend adapter that implements the PhysicsBackend interface.

This wraps the original MPM_Simulator_WARP class and exposes a clean,
backend-agnostic API through PhysicsBackend / SimulationState.

Normalization note: Warp-MPM requires positions in a fixed [0, grid_lim]^3
domain.  All coordinate normalization (transform2origin, shift2center111,
covariance scaling, particle filling, volume computation) is performed
*inside* this backend so that the pipeline can operate entirely in rotated
world space.
"""

import numpy as np
import warp as wp
import torch

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.warp_mpm.mpm_solver_warp import MPM_Simulator_WARP
from physics_sim.backend.newton_mpm.kernels import compute_R_quats_scales_from_F

from physics_sim.preprocessing.transform import (
    transform2origin,
    transform_with_reference,
    shift2center111,
    undoshift2center111,
    undotransform2origin,
)

try:
    from physics_sim.preprocessing.particle_filling import (
        fill_particles as _fill_particles,
        get_particle_volume as _get_particle_volume,
    )
except ImportError:
    _fill_particles = None
    _get_particle_volume = None


def _set_boundary_conditions(mpm_solver: MPM_Simulator_WARP, bc_params: list, time_params: dict):
    """Register boundary conditions on the raw MPM solver.

    Copied from the original PhysGaussian decode_param.set_boundary_conditions,
    kept here so that the warp_mpm backend is fully self-contained.
    """
    for bc in bc_params:
        bc_type = bc["type"]

        if bc_type == "cuboid":
            assert "point" in bc and "size" in bc and "velocity" in bc
            mpm_solver.set_velocity_on_cuboid(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=bc.get("start_time", 0.0),
                end_time=bc.get("end_time", 1e3),
                reset=bc.get("reset", 0),
            )

        elif bc_type == "particle_impulse":
            assert "force" in bc
            mpm_solver.add_impulse_on_particles(
                force=bc["force"],
                dt=time_params["substep_dt"],
                point=bc.get("point", [1, 1, 1]),
                size=bc.get("size", [1, 1, 1]),
                num_dt=bc.get("num_dt", 1),
                start_time=bc.get("start_time", 0.0),
            )

        elif bc_type == "bounding_box":
            mpm_solver.add_bounding_box()

        elif bc_type == "enforce_particle_translation":
            assert all(k in bc for k in ("point", "size", "velocity", "start_time", "end_time"))
            mpm_solver.enforce_particle_velocity_translation(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        elif bc_type == "surface_collider":
            assert all(k in bc for k in ("point", "normal", "surface", "friction", "start_time", "end_time"))
            mpm_solver.add_surface_collider(
                point=bc["point"],
                normal=bc["normal"],
                surface=bc["surface"],
                friction=bc["friction"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        elif bc_type == "release_particles_sequentially":
            assert all(k in bc for k in ("normal", "start_position", "end_position", "num_layers", "start_time", "end_time"))
            mpm_solver.release_particles_sequentially(
                normal=bc["normal"],
                start_position=bc["start_position"],
                end_position=bc["end_position"],
                num_layers=bc["num_layers"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        elif bc_type == "enforce_particle_velocity_rotation":
            assert all(k in bc for k in ("normal", "point", "start_time", "end_time", "half_height_and_radius", "rotation_scale", "translation_scale"))
            mpm_solver.enforce_particle_velocity_rotation(
                point=bc["point"],
                normal=bc["normal"],
                half_height_and_radius=bc["half_height_and_radius"],
                rotation_scale=bc["rotation_scale"],
                translation_scale=bc["translation_scale"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        else:
            raise TypeError(f"Undefined BC type: {bc_type}")


class WarpMPMBackend(PhysicsBackend):
    """Physics backend powered by Warp-MPM (Material Point Method).

    Unlike Newton-based backends, Warp-MPM operates in a normalised
    [0, grid_lim]^3 domain.  This class transparently handles:

    * Forward normalisation in ``initialize()`` (positions, covariances)
    * Particle filling and volume computation
    * Inverse normalisation in ``get_state()``
    * BC point conversion in ``set_boundary_conditions()``

    The pipeline only needs to pass rotated world-space data.
    """

    def __init__(self, device: str = "cuda:0"):
        self._solver: MPM_Simulator_WARP = None
        self._device = device

        # Normalisation bookkeeping (set in initialize)
        self._scale_origin: torch.Tensor | None = None
        self._original_mean_pos: torch.Tensor | None = None

        # 2DGS support
        self._init_quats_wp: wp.array | None = None
        self._init_scales_wp: wp.array | None = None
        self._out_quats_wp: wp.array | None = None
        self._out_scales_wp: wp.array | None = None
        self._num_scales: int = 0
        self._out_R_wp: wp.array | None = None

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
        """Build and load the Warp-MPM solver from rotated world-space data.

        The method internally normalises positions to [0, grid_lim]^3,
        optionally fills interior particles, and computes per-particle
        volumes.  The caller does NOT need to pre-normalise.

        Extra kwargs consumed here (ignored by other backends):
            scale (float):  Preprocessing scale factor (default 1.0).
            opacity (Tensor | None):  (N, 1) per-particle opacity for filling.
            filling_params (dict | None):  Particle filling configuration.
            material_grid_lim (float):  grid_lim used for fill dx (default grid_lim).
        """
        scale = kwargs.get("scale", 1.0)
        opacity = kwargs.get("opacity")
        filling_params = kwargs.get("filling_params")

        # ── 1. Normalise positions to MPM domain ─────────────────────
        transformed_pos, scale_origin, original_mean_pos = transform2origin(
            positions, scale
        )
        transformed_pos = shift2center111(transformed_pos)

        self._scale_origin = scale_origin
        self._original_mean_pos = original_mean_pos

        # ── 2. Scale covariances ─────────────────────────────────────
        scaled_cov = (scale_origin * scale_origin * covariances).to(self._device)

        # ── 3. Particle filling (optional) ───────────────────────────
        gs_num = transformed_pos.shape[0]
        if filling_params is not None and opacity is not None:
            if _fill_particles is None:
                raise ModuleNotFoundError(
                    "Particle filling requires Taichi (physics_sim.preprocessing."
                    "particle_filling).  Install taichi or disable particle_filling."
                )
            fill_grid_dx = grid_lim / filling_params["n_grid"]
            mpm_pos = _fill_particles(
                pos=transformed_pos,
                opacity=opacity,
                cov=scaled_cov[:gs_num],
                grid_n=filling_params["n_grid"],
                max_samples=filling_params["max_particles_num"],
                grid_dx=fill_grid_dx,
                density_thres=filling_params["density_threshold"],
                search_thres=filling_params["search_threshold"],
                max_particles_per_cell=filling_params["max_particles_per_cell"],
                search_exclude_dir=filling_params["search_exclude_direction"],
                ray_cast_dir=filling_params["ray_cast_direction"],
                boundary=filling_params["boundary"],
                smooth=filling_params["smooth"],
            ).to(device=self._device)
            print(f"[WarpMPM] Filled: {gs_num} gs + {mpm_pos.shape[0] - gs_num} interior")
        else:
            mpm_pos = transformed_pos.to(device=self._device)

        # ── 4. Compute volumes ───────────────────────────────────────
        if _get_particle_volume is not None:
            dx = grid_lim / n_grid
            mpm_vol = _get_particle_volume(
                mpm_pos, n_grid, dx, uniform=False,
            ).to(device=self._device)
        else:
            dx = grid_lim / n_grid
            mpm_vol = torch.full(
                (mpm_pos.shape[0],), float(dx ** 3), device=self._device,
            )

        # ── 5. Build full covariance array (gs + filled zeros) ───────
        mpm_cov = torch.zeros((mpm_pos.shape[0], 6), device=self._device)
        mpm_cov[:gs_num] = scaled_cov[:gs_num]

        # ── 6. Create raw solver ─────────────────────────────────────
        self._solver = MPM_Simulator_WARP(10)
        self._solver.load_initial_data_from_torch(
            mpm_pos, mpm_vol, mpm_cov,
            n_grid=n_grid, grid_lim=grid_lim, device=self._device,
        )

        # ── 7. 2DGS quats / scales ──────────────────────────────────
        iq = kwargs.get("init_quats")
        isc = kwargs.get("init_scales")
        if iq is not None and isc is not None:
            n = mpm_pos.shape[0]
            self._num_scales = isc.shape[1]
            iq_np = iq.detach().cpu().numpy().astype(np.float32)
            isc_np = isc.detach().cpu().numpy().astype(np.float32)
            self._init_quats_wp = wp.from_numpy(
                iq_np.reshape(-1, 4), dtype=wp.vec4, device=self._device,
            )
            self._init_scales_wp = wp.from_numpy(
                isc_np.reshape(-1), dtype=float, device=self._device,
            )
            self._out_quats_wp = wp.zeros(n, dtype=wp.vec4, device=self._device)
            self._out_scales_wp = wp.zeros(
                n * self._num_scales, dtype=float, device=self._device,
            )
            self._out_R_wp = wp.zeros(n, dtype=wp.mat33, device=self._device)

    def set_material(self, material_params: dict) -> None:
        self._solver.set_parameters_dict(material_params, device=self._device)
        self._material_set = True

    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
        """Convert BCs from rotated world space to MPM space, then register."""
        converted = []
        for bc in bc_params:
            bc_new = dict(bc)
            # Convert position-valued fields to MPM space
            if "point" in bc and bc.get("type") != "bounding_box":
                p = torch.tensor(bc["point"], device="cuda", dtype=torch.float32).reshape(1, 3)
                p_mpm = shift2center111(
                    transform_with_reference(p, self._scale_origin, self._original_mean_pos)
                )[0]
                bc_new["point"] = [float(x) for x in p_mpm.cpu().tolist()]
            if "size" in bc:
                s = torch.tensor(bc["size"], device="cuda", dtype=torch.float32)
                s_mpm = s * self._scale_origin
                bc_new["size"] = [float(x) for x in s_mpm.cpu().tolist()]
            if "start_position" in bc:
                p = torch.tensor(bc["start_position"], device="cuda", dtype=torch.float32).reshape(1, 3)
                p_mpm = shift2center111(
                    transform_with_reference(p, self._scale_origin, self._original_mean_pos)
                )[0]
                bc_new["start_position"] = [float(x) for x in p_mpm.cpu().tolist()]
            if "end_position" in bc:
                p = torch.tensor(bc["end_position"], device="cuda", dtype=torch.float32).reshape(1, 3)
                p_mpm = shift2center111(
                    transform_with_reference(p, self._scale_origin, self._original_mean_pos)
                )[0]
                bc_new["end_position"] = [float(x) for x in p_mpm.cpu().tolist()]
            converted.append(bc_new)
        _set_boundary_conditions(self._solver, converted, time_params)

    def finalize(self) -> None:
        """Finalize material parameters (mu, lam). Must be called after set_boundary_conditions."""
        self._solver.finalize_mu_lam(device=self._device)

    def step(self, dt: float, frame: int) -> None:
        self._solver.p2g2p(frame, dt, device=self._device)

    def get_state(self) -> SimulationState:
        """Export state with positions/covariances converted back to rotated world space."""
        pos = self._solver.export_particle_x_to_torch()
        cov = self._solver.export_particle_cov_to_torch(device=self._device)
        vel = self._solver.export_particle_v_to_torch()

        # ── Inverse normalisation: MPM → rotated world space ─────────
        pos = undotransform2origin(
            undoshift2center111(pos),
            self._scale_origin,
            self._original_mean_pos,
        )
        s2 = self._scale_origin * self._scale_origin
        cov = cov.view(-1, 6) / s2

        out_quats_t = None
        out_scales_t = None

        if self._init_quats_wp is not None:
            n = self._solver.n_particles
            F = self._solver.mpm_state.particle_F_trial
            wp.launch(
                compute_R_quats_scales_from_F,
                dim=n,
                inputs=[
                    F,
                    self._init_quats_wp,
                    self._init_scales_wp,
                    self._num_scales,
                    self._out_R_wp,
                    self._out_quats_wp,
                    self._out_scales_wp,
                ],
                device=self._device,
            )
            rot = wp.to_torch(self._out_R_wp).reshape(-1, 9)
            out_quats_t = wp.to_torch(self._out_quats_wp).view(n, 4)
            out_scales_t = wp.to_torch(self._out_scales_wp).view(n, self._num_scales)
        else:
            rot = self._solver.export_particle_R_to_torch(device=self._device)

        return SimulationState(
            positions=pos,
            covariances=cov,
            rotations=rot.view(-1, 3, 3),
            velocities=vel,
            quats=out_quats_t,
            scales=out_scales_t,
        )

    # ------------------------------------------------------------------
    # Extra accessors (for advanced users who need raw solver access)
    # ------------------------------------------------------------------

    @property
    def raw_solver(self) -> MPM_Simulator_WARP:
        """Escape hatch: direct access to the underlying MPM solver."""
        return self._solver
