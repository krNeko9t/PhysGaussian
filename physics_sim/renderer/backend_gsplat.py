"""Rasterization backend using gsplat (nerfstudio-project/gsplat)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from physics_sim.renderer.backend_base import RasterBackend

if TYPE_CHECKING:
    from physics_sim.renderer.gs_renderer import SimpleCamera


def _cov6_to_mat3(cov6: torch.Tensor) -> torch.Tensor:
    """Convert (N, 6) upper-triangle covariance to (N, 3, 3) symmetric matrix.

    Input order: ``[c00, c01, c02, c11, c12, c22]``.
    """
    N = cov6.shape[0]
    mat = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    mat[:, 0, 0] = cov6[:, 0]
    mat[:, 0, 1] = cov6[:, 1]
    mat[:, 1, 0] = cov6[:, 1]
    mat[:, 0, 2] = cov6[:, 2]
    mat[:, 2, 0] = cov6[:, 2]
    mat[:, 1, 1] = cov6[:, 3]
    mat[:, 1, 2] = cov6[:, 4]
    mat[:, 2, 1] = cov6[:, 4]
    mat[:, 2, 2] = cov6[:, 5]
    return mat


class GsplatBackend(RasterBackend):
    """Wraps ``gsplat.rendering.rasterization`` for 3DGS rendering."""

    def render(
        self,
        camera: SimpleCamera,
        means: torch.Tensor,
        cov6: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        try:
            from gsplat.rendering import rasterization
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "gsplat is required for the 'gsplat' backend. "
                "Install it (pip install gsplat) or use --raster_backend diffrast."
            ) from e

        covars = _cov6_to_mat3(cov6)  # (N, 3, 3)

        viewmats = camera.viewmat.unsqueeze(0)  # (1, 4, 4)
        Ks = camera.K.unsqueeze(0)  # (1, 3, 3)

        opacities_1d = opacities.squeeze(-1)  # (N,)

        backgrounds = None
        if bg_color is not None:
            backgrounds = bg_color # .unsqueeze(0)  # (1, 3)

        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=None,
            scales=None,
            opacities=opacities_1d,
            colors=colors,
            covars=covars,
            viewmats=viewmats,
            Ks=Ks,
            width=int(camera.image_width),
            height=int(camera.image_height),
            near_plane=camera.znear,
            far_plane=camera.zfar,
            backgrounds=backgrounds,
            sh_degree=None,
        )

        # render_colors: (1, H, W, 3) -> (3, H, W)
        rendered = render_colors[0].permute(2, 0, 1)

        meta["render_alphas"] = render_alphas
        return rendered, meta
