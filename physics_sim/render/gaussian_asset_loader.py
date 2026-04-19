"""Gaussian asset loading from PLY files."""

from __future__ import annotations

import numpy as np
import torch
from plyfile import PlyData

from physics_sim.render.gaussian_geometry import build_covariance
from physics_sim.sh_contract import (
    assemble_sh_from_ply_features,
    expected_f_rest_feature_count,
)
from physics_sim.render.types import GaussianAsset


class GaussianAssetLoader:
    """Load 3DGS/2DGS data from PLY into typed tensors."""

    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree

    def load_ply(self, ply_path: str) -> GaussianAsset:
        plydata = PlyData.read(ply_path)
        vtx = plydata.elements[0]

        xyz = np.stack([np.asarray(vtx["x"]), np.asarray(vtx["y"]), np.asarray(vtx["z"])], axis=1)
        opacities = np.asarray(vtx["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vtx["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vtx["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vtx["f_dc_2"])

        extra_f_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        expected_rest = expected_f_rest_feature_count(self.sh_degree)
        assert len(extra_f_names) == expected_rest, (
            f"Expected {expected_rest} f_rest features for SH degree {self.sh_degree}, "
            f"got {len(extra_f_names)}"
        )
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(vtx[attr_name])
        shs_np = assemble_sh_from_ply_features(
            features_dc=features_dc,
            features_rest=features_extra,
            sh_degree=self.sh_degree,
        )

        scale_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        num_scales = len(scale_names)
        gs_type = "2dgs" if num_scales == 2 else "3dgs"

        scales = np.zeros((xyz.shape[0], num_scales))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vtx[attr_name])

        rot_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vtx[attr_name])

        scales_for_cov = scales
        if num_scales == 2:
            scales_for_cov = np.concatenate(
                [scales, np.full((xyz.shape[0], 1), -16.0, dtype=np.float32)],
                axis=1,
            )

        pos = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        opacity = torch.sigmoid(torch.tensor(opacities, dtype=torch.float32, device="cuda"))
        scaling = torch.exp(torch.tensor(scales_for_cov, dtype=torch.float32, device="cuda"))
        rotation_quat = torch.tensor(rots, dtype=torch.float32, device="cuda")
        rotation_quat = torch.nn.functional.normalize(rotation_quat, dim=1)

        scales_raw = torch.exp(torch.tensor(scales, dtype=torch.float32, device="cuda"))
        shs = torch.tensor(shs_np, dtype=torch.float32, device="cuda")

        cov3D = build_covariance(scaling, rotation_quat)
        screen_points = torch.zeros_like(pos, requires_grad=False)
        return GaussianAsset(
            pos=pos,
            cov3D_precomp=cov3D,
            opacity=opacity,
            shs=shs,
            screen_points=screen_points,
            gs_type=gs_type,
            quats=rotation_quat,
            scales=scales_raw,
        )
