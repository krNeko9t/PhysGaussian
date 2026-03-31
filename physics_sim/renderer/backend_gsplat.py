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
    """Wraps ``gsplat.rendering.rasterization`` for 3DGS and 2DGS."""

    def render(
        self,
        camera: SimpleCamera,
        means: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: torch.Tensor | None = None,
        *,
        cov6: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if quats is not None and scales is not None:
            return self._render_2dgs(camera, means, quats, scales,
                                     colors, opacities, bg_color)
        if cov6 is not None:
            return self._render_3dgs(camera, means, cov6,
                                     colors, opacities, bg_color)
        raise ValueError("GsplatBackend.render requires either cov6 or quats+scales.")

    # ── 3DGS path (covars mode) ──────────────────────────────────────

    def _render_3dgs(
        self,
        camera: SimpleCamera,
        means: torch.Tensor,
        cov6: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        from gsplat.rendering import rasterization

        covars = _cov6_to_mat3(cov6)
        viewmats = camera.viewmat.unsqueeze(0)
        Ks = camera.K.unsqueeze(0)
        opacities_1d = opacities.squeeze(-1)

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
            backgrounds=bg_color,
            sh_degree=None,
        )

        rendered = render_colors[0].permute(2, 0, 1)
        meta["render_alphas"] = render_alphas
        return rendered, meta

    # ── 2DGS path (quats + scales mode) ──────────────────────────────

    def _render_2dgs(
        self,
        camera: SimpleCamera,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        from gsplat.rendering import rasterization_2dgs

        viewmats = camera.viewmat.unsqueeze(0)
        Ks = camera.K.unsqueeze(0)
        opacities_1d = opacities.squeeze(-1)

        render_colors, render_alphas, _, meta = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities_1d,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(camera.image_width),
            height=int(camera.image_height),
            near_plane=camera.znear,
            far_plane=camera.zfar,
            backgrounds=bg_color,
        )

        rendered = render_colors[0].permute(2, 0, 1)
        meta["render_alphas"] = render_alphas
        return rendered, meta
