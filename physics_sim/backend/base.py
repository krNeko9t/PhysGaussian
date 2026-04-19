"""
Abstract base class for physics simulation backends.

Any new backend (e.g. Newton-PBD) only needs to implement PhysicsBackend
and return SimulationState from get_state().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SimulationState:
    """Unified output of a physics simulation step.

    All fields are plain PyTorch tensors so that the rest of the pipeline
    (inverse-preprocessing, rendering) is backend-agnostic.

    ``quats`` and ``scales`` are optional outputs used by 2DGS rendering.
    When the scene is 3DGS these remain None and the renderer uses
    ``covariances`` instead.
    """
    positions: torch.Tensor        # (N, 3) particle positions
    covariances: torch.Tensor      # (N, 6) upper-triangle covariance
    rotations: torch.Tensor        # (N, 3, 3) rotation matrices
    velocities: Optional[torch.Tensor] = None  # (N, 3) optional
    quats: Optional[torch.Tensor] = None       # (N, 4) for 2DGS rendering
    scales: Optional[torch.Tensor] = None      # (N, 2|3) for 2DGS rendering


class PhysicsBackend(ABC):
    """Interface that every physics backend must implement."""

    @abstractmethod
    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        **kwargs,
    ) -> None:
        """Load initial particle data into the backend.

        Args:
            positions:   (N, 3) initial particle positions.
            volumes:     (N,)   per-particle volumes.
            covariances: (N, 6) upper-triangle covariance matrices.
            **kwargs:    backend-specific options (e.g. n_grid, grid_lim).
        """
        ...

    @abstractmethod
    def set_material(self, material_params: dict) -> None:
        """Configure material parameters (Young's modulus, Poisson ratio, etc.)."""
        ...

    @abstractmethod
    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
        """Register boundary conditions for the simulation."""
        ...

    @abstractmethod
    def step(self, dt: float, frame: int) -> None:
        """Advance the simulation by one substep of duration *dt*."""
        ...

    def finalize(self) -> None:
        """Optional post-configuration step (e.g. create solver, compile).

        Called after set_material() and set_boundary_conditions().
        Subclasses may override; default is a no-op.
        """
        pass

    @abstractmethod
    def get_state(self) -> SimulationState:
        """Export the current simulation state as PyTorch tensors.

        The returned SimulationState must contain *all* particles
        (original + filled), so the caller can slice [:gs_num] as needed.
        """
        ...

    def get_diagnostics(self) -> Optional[dict]:
        """Optional diagnostics hook consumed by the stage loop.

        Backends that do not expose runtime diagnostics should keep
        the default ``None`` implementation.
        """
        return None
