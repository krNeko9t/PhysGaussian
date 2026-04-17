"""SH-to-RGB color conversion for gaussian rendering."""

from __future__ import annotations

from typing import Optional

import torch

from physics_sim.render.gaussian_math import eval_sh


class ShColorizer:
    """Convert spherical harmonics features into precomputed RGB colors."""

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree

    def convert_sh(
        self,
        shs: torch.Tensor,
        camera,
        position: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
        alignment_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shs_view = shs.transpose(1, 2).view(-1, 3, (self.sh_degree + 1) ** 2)
        dir_pp = position - camera.camera_center.repeat(shs_view.shape[0], 1)
        if rotation is not None:
            n = rotation.shape[0]
            dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].unsqueeze(2)).squeeze(2)
        if alignment_inv is not None:
            dir_pp = torch.matmul(dir_pp, alignment_inv.T)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.sh_degree, shs_view, dir_pp_normalized)
        return torch.clamp_min(sh2rgb + 0.5, 0.0)
