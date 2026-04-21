"""No-op physics backend (backend_type == "none").

Returns the initial state unchanged on every step.  Useful for
render-only tests or debugging the pipeline without physics.
"""

from __future__ import annotations

from typing import Optional

import torch

from physics_sim.backend.base import PhysicsBackend, SimulationState
from physics_sim.backend.spec import MaterialSetupSpec
from physics_sim.config.models import BoundaryCondition, NoneBackendConfig, TimeConfig


class NoneBackend(PhysicsBackend):

    def __init__(self, *, cfg: NoneBackendConfig, device: str = "cuda:0"):
        self._cfg = cfg
        self._device = device
        self._pos: Optional[torch.Tensor] = None
        self._cov: Optional[torch.Tensor] = None
        self._rot: Optional[torch.Tensor] = None
        self._quats: Optional[torch.Tensor] = None
        self._scales: Optional[torch.Tensor] = None

    def initialize(
        self,
        positions: torch.Tensor,
        volumes: torch.Tensor,
        covariances: torch.Tensor,
        *,
        init_quats: Optional[torch.Tensor] = None,
        init_scales: Optional[torch.Tensor] = None,
    ) -> None:
        self._pos = positions.clone()
        self._cov = covariances.clone()
        n = positions.shape[0]
        self._rot = (
            torch.eye(3, device=self._device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(n, 1, 1)
        )
        self._quats = init_quats.clone() if init_quats is not None else None
        self._scales = init_scales.clone() if init_scales is not None else None

    def set_material(self, spec: MaterialSetupSpec) -> None:
        pass

    def set_boundary_conditions(
        self,
        bcs: list[BoundaryCondition],
        time: TimeConfig,
    ) -> None:
        pass

    def step(self, dt: float, frame: int) -> None:
        pass

    def get_state(self) -> SimulationState:
        return SimulationState(
            positions=self._pos,
            covariances=self._cov,
            rotations=self._rot,
            quats=self._quats,
            scales=self._scales,
        )
