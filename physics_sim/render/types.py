"""Typed data models for render-domain boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianAsset:
    """Loaded Gaussian asset tensors shared by scene and render stages."""

    pos: torch.Tensor
    cov3D_precomp: torch.Tensor
    opacity: torch.Tensor
    shs: torch.Tensor  # (N, C, 3), C=(sh_degree+1)^2
    screen_points: torch.Tensor
    gs_type: str
    quats: torch.Tensor
    scales: torch.Tensor

    def to_dict(self) -> dict:
        """Compatibility helper for legacy callsites expecting dict payloads."""
        return {
            "pos": self.pos,
            "cov3D_precomp": self.cov3D_precomp,
            "opacity": self.opacity,
            "shs": self.shs,
            "screen_points": self.screen_points,
            "gs_type": self.gs_type,
            "quats": self.quats,
            "scales": self.scales,
        }
