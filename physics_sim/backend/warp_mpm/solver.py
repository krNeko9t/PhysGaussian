"""
Warp-MPM backend adapter that implements the PhysicsBackend interface.

This wraps the original MPM_Simulator_WARP class and exposes a clean,
backend-agnostic API through PhysicsBackend / SimulationState.
"""

import warp as wp
import torch

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.warp_mpm.mpm_solver_warp import MPM_Simulator_WARP


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
    """Physics backend powered by Warp-MPM (Material Point Method)."""

    def __init__(self, device: str = "cuda:0"):
        self._solver: MPM_Simulator_WARP = None
        self._device = device

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
        self._solver = MPM_Simulator_WARP(10)
        self._solver.load_initial_data_from_torch(
            positions, volumes, covariances,
            n_grid=n_grid, grid_lim=grid_lim, device=self._device,
        )

    def set_material(self, material_params: dict) -> None:
        self._solver.set_parameters_dict(material_params, device=self._device)
        # NOTE: finalize_mu_lam must be called AFTER set_boundary_conditions
        # (same order as original gs_simulation.py). Call finalize() separately.
        self._material_set = True

    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
        _set_boundary_conditions(self._solver, bc_params, time_params)

    def finalize(self) -> None:
        """Finalize material parameters (mu, lam). Must be called after set_boundary_conditions."""
        self._solver.finalize_mu_lam(device=self._device)

    def step(self, dt: float, frame: int) -> None:
        self._solver.p2g2p(frame, dt, device=self._device)

    def get_state(self) -> SimulationState:
        pos = self._solver.export_particle_x_to_torch()
        cov = self._solver.export_particle_cov_to_torch(device=self._device)
        rot = self._solver.export_particle_R_to_torch(device=self._device)
        return SimulationState(
            positions=pos,
            covariances=cov.view(-1, 6),
            rotations=rot.view(-1, 3, 3),
            velocities=self._solver.export_particle_v_to_torch(),
        )

    # ------------------------------------------------------------------
    # Extra accessors (for advanced users who need raw solver access)
    # ------------------------------------------------------------------

    @property
    def raw_solver(self) -> MPM_Simulator_WARP:
        """Escape hatch: direct access to the underlying MPM solver."""
        return self._solver
