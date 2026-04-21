"""
Abstract base class for physics simulation backends.

Any new backend (e.g. Newton-PBD) only needs to implement PhysicsBackend
and return SimulationState from get_state().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from physics_sim.backend.spec import MaterialSetupSpec
from physics_sim.config.models import BoundaryCondition, TimeConfig


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
    """Interface that every physics backend must implement.

    Solver numerical options (``n_grid``, ``grid_lim``,
    ``solver_iterations``, etc.) are consumed directly from the typed
    ``BackendConfig`` passed to each backend's constructor — they do not
    flow through this interface.
    """

    @abstractmethod
    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        *,
        init_quats: Optional[torch.Tensor] = None,
        init_scales: Optional[torch.Tensor] = None,
    ) -> None:
        """Load initial particle data into the backend.

        ``init_quats`` / ``init_scales`` are only used by 2DGS scenes.
        """
        ...

    @abstractmethod
    def set_material(self, spec: MaterialSetupSpec) -> None:
        """Configure gravity and per-object material parameters."""
        ...

    @abstractmethod
    def set_boundary_conditions(
        self,
        bcs: list[BoundaryCondition],
        time: TimeConfig,
    ) -> None:
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
        """Optional diagnostics hook consumed by the stage loop."""
        return None
