"""No-op physics backend (backend_type == "none").

Returns the initial state unchanged on every step.  Useful for
render-only tests or debugging the pipeline without physics.
"""

from __future__ import annotations

from typing import Optional

import torch

from physics_sim.backend.base import PhysicsBackend, SimulationState


class NoneBackend(PhysicsBackend):

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._pos: Optional[torch.Tensor] = None
        self._cov: Optional[torch.Tensor] = None
        self._rot: Optional[torch.Tensor] = None
        self._quats: Optional[torch.Tensor] = None
        self._scales: Optional[torch.Tensor] = None

    def initialize(self, positions, volumes, covariances, **kwargs):
        self._pos = positions.clone()
        self._cov = covariances.clone()
        n = positions.shape[0]
        self._rot = (
            torch.eye(3, device=self._device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(n, 1, 1)
        )
        iq = kwargs.get("init_quats")
        isc = kwargs.get("init_scales")
        self._quats = iq.clone() if iq is not None else None
        self._scales = isc.clone() if isc is not None else None

    def set_material(self, material_params: dict) -> None:
        pass

    def set_boundary_conditions(self, bc_params: list, time_params: dict) -> None:
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
