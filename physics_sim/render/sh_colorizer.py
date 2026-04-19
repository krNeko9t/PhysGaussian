"""SH-to-RGB color conversion for gaussian rendering."""

from __future__ import annotations

from typing import Optional

import torch

from physics_sim.render.camera import SimpleCamera
from physics_sim.render.sh_eval import eval_sh
from physics_sim.sh_contract import sh_coeff_count


def compute_view_directions(
    position: torch.Tensor,
    camera_center: torch.Tensor,
    view_rotations: Optional[torch.Tensor] = None,
    alignment_inv: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute normalized view directions in SH source space."""
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"position must have shape (N, 3), got {tuple(position.shape)}.")
    if camera_center.ndim != 1 or camera_center.shape[0] != 3:
        raise ValueError(
            f"camera_center must have shape (3,), got {tuple(camera_center.shape)}."
        )
    dirs = position - camera_center.unsqueeze(0)
    if view_rotations is not None:
        if view_rotations.shape != (position.shape[0], 3, 3):
            raise ValueError(
                "view_rotations must have shape "
                f"({position.shape[0]}, 3, 3), got {tuple(view_rotations.shape)}."
            )
        dirs = torch.matmul(view_rotations, dirs.unsqueeze(2)).squeeze(2)
    if alignment_inv is not None:
        if alignment_inv.shape != (3, 3):
            raise ValueError(f"alignment_inv must have shape (3, 3), got {alignment_inv.shape}.")
        dirs = torch.matmul(dirs, alignment_inv.T)
    return dirs / dirs.norm(dim=1, keepdim=True).clamp_min(eps)


class ShColorizer:
    """Convert spherical harmonics features into precomputed RGB colors."""

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree

    def convert_sh(
        self,
        shs: torch.Tensor,
        camera: SimpleCamera,
        position: torch.Tensor,
        *,
        view_rotations: Optional[torch.Tensor] = None,
        alignment_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coeffs = sh_coeff_count(self.sh_degree)
        if shs.ndim != 3 or shs.shape[1:] != (coeffs, 3):
            raise ValueError(f"shs must have shape (N, {coeffs}, 3), got {tuple(shs.shape)}.")
        shs_view = shs.transpose(1, 2).reshape(-1, 3, coeffs)
        view_dirs = compute_view_directions(
            position=position,
            camera_center=camera.camera_center,
            view_rotations=view_rotations,
            alignment_inv=alignment_inv,
        )
        sh2rgb = eval_sh(self.sh_degree, shs_view, view_dirs)
        return torch.clamp_min(sh2rgb + 0.5, 0.0)
