"""
Self-contained 3DGS renderer for physics_sim.

This module removes the dependency on the gaussian-splatting training codebase
by reimplementing just the subset of logic needed for rendering:
  - PLY loading (positions, SH features, scaling, rotation, opacity)
  - Covariance computation from scaling + rotation
  - Spherical-harmonics → RGB conversion
  - Rasterisation via pluggable backend (gsplat or diff_gaussian_rasterization)
  - Camera construction from cameras.json

Rasterization is delegated to a ``RasterBackend`` (see ``backend_base.py``).
Supported backends: ``gsplat`` (default), ``diffrast``.
"""

import math
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData



# ── SH constants (from PlenOctree) ───────────────────────────────────────

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]
C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601,
      -0.6690465435572892, 0.10578554691520431, -0.6690465435572892,
      0.47308734787878004, -1.7701307697799304, 0.6258357354491761]


def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics at unit directions (up to degree 4)."""
    assert 0 <= deg <= 4
    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2]
                  - C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                      + C2[0] * xy * sh[..., 4]
                      + C2[1] * yz * sh[..., 5]
                      + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                      + C2[3] * xz * sh[..., 7]
                      + C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result
                          + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                          + C3[1] * xy * z * sh[..., 10]
                          + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                          + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                          + C3[5] * z * (xx - yy) * sh[..., 14]
                          + C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result
                              + C4[0] * xy * (xx - yy) * sh[..., 16]
                              + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                              + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                              + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                              + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                              + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                              + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                              + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                              + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


# ── Gaussian helpers ──────────────────────────────────────────────────────

def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Quaternion (N,4) → rotation matrix (N,3,3)."""
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    rr, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.size(0), 3, 3), device="cuda")
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - rr * z)
    R[:, 0, 2] = 2 * (x * z + rr * y)
    R[:, 1, 0] = 2 * (x * y + rr * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - rr * x)
    R[:, 2, 0] = 2 * (x * z - rr * y)
    R[:, 2, 1] = 2 * (y * z + rr * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_covariance(scaling: torch.Tensor, rotation_quat: torch.Tensor,
                     scaling_modifier: float = 1.0) -> torch.Tensor:
    """Compute the upper-triangle (N,6) covariance from scaling + rotation."""
    L = torch.zeros((scaling.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(rotation_quat)
    L[:, 0, 0] = scaling_modifier * scaling[:, 0]
    L[:, 1, 1] = scaling_modifier * scaling[:, 1]
    L[:, 2, 2] = scaling_modifier * scaling[:, 2]
    L = R @ L
    cov = L @ L.transpose(1, 2)
    # upper triangle
    out = torch.zeros((cov.shape[0], 6), dtype=torch.float, device="cuda")
    out[:, 0] = cov[:, 0, 0]
    out[:, 1] = cov[:, 0, 1]
    out[:, 2] = cov[:, 0, 2]
    out[:, 3] = cov[:, 1, 1]
    out[:, 4] = cov[:, 1, 2]
    out[:, 5] = cov[:, 2, 2]
    return out


# ── Camera helpers ────────────────────────────────────────────────────────

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Inverse of focal2fov. fov in radians."""
    return pixels / (2.0 * math.tan(fov * 0.5))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class SimpleCamera:
    """Minimal camera that mirrors the interface expected by the rasterizer."""

    def __init__(self, R, T, FoVx, FoVy, width, height):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height
        self.znear = 0.01
        self.zfar = 100.0

        w2v = getWorld2View2(R, T)

        # Row-major world-to-camera 4x4 (used by gsplat)
        self.viewmat = torch.tensor(w2v, dtype=torch.float32).cuda()

        # Column-major variant (used by diff_gaussian_rasterization)
        self.world_view_transform = self.viewmat.transpose(0, 1).contiguous()

        self.projection_matrix = (
            getProjectionMatrix(self.znear, self.zfar, FoVx, FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            ).squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # 3x3 intrinsic matrix (used by gsplat)
        fx = fov2focal(FoVx, float(width))
        fy = fov2focal(FoVy, float(height))
        self.K = torch.tensor([
            [fx, 0.0, width * 0.5],
            [0.0, fy, height * 0.5],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32).cuda()


def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(camera_to_object)
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)
    first_column = np.cross(camera_y, camera_to_object)
    return np.column_stack((first_column, camera_y, camera_to_object))


def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical = (
        np.array([
            np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
            np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
            np.sin(elevation / 180.0 * np.pi),
        ]) * radius
    )
    return center + observant_coordinates @ canonical


def get_camera_position_and_rotation(azimuth, elevation, radius, view_center, observant_coordinates):
    position = get_point_on_sphere(azimuth, elevation, radius, view_center, observant_coordinates)
    R = generate_camera_rotation_matrix(view_center - position, -observant_coordinates[:, 2])
    return position, R


def get_current_radius_azimuth_and_elevation(camera_position, view_center, observant_coordinates):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observant_coordinates[:, 2])
    cosine = dot_product / (np.linalg.norm(center2camera) * np.linalg.norm(observant_coordinates[:, 2]))
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observant_coordinates[:, 0])
    cosine2 = dot_product2 / (np.linalg.norm(proj_onto_hori) * np.linalg.norm(observant_coordinates[:, 0]))
    if np.dot(proj_onto_hori, observant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation


# ── Main renderer class ──────────────────────────────────────────────────

class GaussianRenderer:
    """Self-contained 3DGS renderer that works with a single PLY file.

    Usage::

        renderer = GaussianRenderer(sh_degree=3, raster_backend="gsplat")
        params = renderer.load_ply("path/to/point_cloud.ply")

        cam = renderer.build_camera_from_json("cameras.json", camera_params, ...)
        colors = renderer.convert_sh(shs, cam, pos, rot)
        image, meta = renderer.render(cam, pos, cov3D, colors, opacity, bg)
    """

    def __init__(self, sh_degree: int = 3, raster_backend: str = "gsplat"):
        from physics_sim.renderer.backend_base import create_raster_backend

        self.sh_degree = sh_degree
        self.backend = create_raster_backend(raster_backend, sh_degree)

    def load_ply(self, ply_path: str) -> dict:
        """Load a 3DGS or 2DGS PLY checkpoint and return extracted parameters.

        The format is auto-detected from the number of ``scale_*`` properties:
        3 scales -> 3DGS, 2 scales -> 2DGS.

        Returns a dict with keys:
            pos            (N, 3) float32  – positions
            cov3D_precomp  (N, 6) float32  – upper-triangle covariance
            opacity        (N, 1) float32  – activated opacity (sigmoid)
            shs            (N, K, 3) float32 – spherical harmonics features
            screen_points  (N, 3) float32  – zeros placeholder for 2D means
            gs_type        str  – ``"3dgs"`` or ``"2dgs"``
            quats          (N, 4) float32  – normalised quaternion rotations
            scales         (N, 2) or (N, 3) float32  – activated scales (exp)
        """
        plydata = PlyData.read(ply_path)
        vtx = plydata.elements[0]

        xyz = np.stack([np.asarray(vtx["x"]), np.asarray(vtx["y"]), np.asarray(vtx["z"])], axis=1)
        opacities = np.asarray(vtx["opacity"])[..., np.newaxis]

        # DC features
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vtx["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vtx["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vtx["f_dc_2"])

        # rest SH features
        extra_f_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        expected_rest = 3 * (self.sh_degree + 1) ** 2 - 3
        assert len(extra_f_names) == expected_rest, (
            f"Expected {expected_rest} f_rest features for SH degree {self.sh_degree}, "
            f"got {len(extra_f_names)}"
        )
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(vtx[attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1)
        )

        # scaling & rotation
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

        # For 2DGS, pad a near-zero third scale so build_covariance (which
        # needs 3 scales) produces a valid thin-disc covariance for physics.
        scales_for_cov = scales
        if num_scales == 2:
            scales_for_cov = np.concatenate(
                [scales, np.full((xyz.shape[0], 1), -16.0, dtype=np.float32)],
                axis=1,
            )

        # Convert to CUDA tensors
        pos = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        opacity = torch.sigmoid(torch.tensor(opacities, dtype=torch.float32, device="cuda"))
        scaling = torch.exp(torch.tensor(scales_for_cov, dtype=torch.float32, device="cuda"))
        rotation_quat = torch.tensor(rots, dtype=torch.float32, device="cuda")
        rotation_quat = torch.nn.functional.normalize(rotation_quat, dim=1)

        # Original (non-padded) activated scales for rendering
        scales_raw = torch.exp(torch.tensor(scales, dtype=torch.float32, device="cuda"))

        # features: (N, SH_total, 3) – dc first, then rest
        features_dc_t = torch.tensor(features_dc, dtype=torch.float32, device="cuda")  # (N,3,1)
        features_rest_t = torch.tensor(features_extra, dtype=torch.float32, device="cuda")  # (N,3,SH-1)
        shs = torch.cat([features_dc_t, features_rest_t], dim=2)  # (N, 3, SH_total)
        shs = shs.transpose(1, 2)  # -> (N, SH_total, 3)

        cov3D = build_covariance(scaling, rotation_quat)
        screen_points = torch.zeros_like(pos, requires_grad=False)

        return {
            "pos": pos,
            "cov3D_precomp": cov3D,
            "opacity": opacity,
            "shs": shs,
            "screen_points": screen_points,
            "gs_type": gs_type,
            "quats": rotation_quat,
            "scales": scales_raw,
        }

    # ── SH → RGB conversion ───────────────────────────────────────────

    def convert_sh(
        self,
        shs: torch.Tensor,
        camera: "SimpleCamera",
        position: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert spherical harmonics to precomputed RGB colours.

        Args:
            shs:      (N, SH, 3) spherical harmonic coefficients.
            camera:   a SimpleCamera (needs .camera_center).
            position: (N, 3) positions.
            rotation: (M, 3, 3) optional per-particle rotation (M <= N).
        """
        shs_view = shs.transpose(1, 2).view(-1, 3, (self.sh_degree + 1) ** 2)
        dir_pp = position - camera.camera_center.repeat(shs_view.shape[0], 1)
        if rotation is not None:
            n = rotation.shape[0]
            dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].unsqueeze(2)).squeeze(2)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.sh_degree, shs_view, dir_pp_normalized)
        return torch.clamp_min(sh2rgb + 0.5, 0.0)

    # ── Render (delegates to backend) ────────────────────────────────────

    def render(
        self,
        camera: "SimpleCamera",
        means: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg_color: Optional[torch.Tensor] = None,
        *,
        cov6: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Render one frame via the selected rasterization backend.

        Pass **either** ``cov6`` (3DGS) **or** ``quats`` + ``scales`` (2DGS).

        Returns:
            rendered: (3, H, W) float32 rendered image.
            meta: backend-specific metadata dict.
        """
        return self.backend.render(
            camera, means, colors, opacities, bg_color,
            cov6=cov6, quats=quats, scales=scales,
        )

    # ── Camera from cameras.json ──────────────────────────────────────

    def build_camera_from_json(
        self,
        cameras_json_path: str,
        camera_params: dict,
        center_view_world_space=None,
        observant_coordinates=None,
        current_frame: int = 0,
    ) -> "SimpleCamera":
        """Build a SimpleCamera from a cameras.json file and camera_params dict."""
        with open(cameras_json_path) as f:
            data = json.load(f)

        default_idx = camera_params.get("default_camera_index", 0)
        show_hint = camera_params.get("show_hint", False)

        if show_hint:
            if default_idx < 0:
                default_idx = 0
            r, a, e = get_current_radius_azimuth_and_elevation(
                data[default_idx]["position"], center_view_world_space, observant_coordinates
            )
            print(f"Default camera {default_idx} has azimuth={a}, elevation={e}, radius={r}")
            print("Now exit program and set your own input!")
            raise SystemExit(0)

        if default_idx > -1:
            raw_camera = data[default_idx]
        else:
            raw_camera = data[0]
            init_a = camera_params["init_azimuthm"]
            init_e = camera_params["init_elevation"]
            init_r = camera_params["init_radius"]
            assert init_a is not None and init_e is not None and init_r is not None

            if camera_params.get("move_camera", False):
                da = camera_params.get("delta_a", 0) or 0
                de = camera_params.get("delta_e", 0) or 0
                dr = camera_params.get("delta_r", 0) or 0
                position, R = get_camera_position_and_rotation(
                    init_a + current_frame * da,
                    init_e + current_frame * de,
                    init_r + current_frame * dr,
                    center_view_world_space, observant_coordinates,
                )
            else:
                position, R = get_camera_position_and_rotation(
                    init_a, init_e, init_r,
                    center_view_world_space, observant_coordinates,
                )
            raw_camera["rotation"] = R.tolist()
            raw_camera["position"] = position.tolist()

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = raw_camera["rotation"]
        tmp[:3, 3] = raw_camera["position"]
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]

        width = raw_camera["width"]
        height = raw_camera["height"]
        fovx = focal2fov(raw_camera["fx"], width)
        fovy = focal2fov(raw_camera["fy"], height)

        return SimpleCamera(R=R, T=T, FoVx=fovx, FoVy=fovy, width=width, height=height)

    # ── Procedural cameras (no cameras.json required) ─────────────────────

    def build_camera_orbit(
        self,
        *,
        camera_params: dict,
        center_view_world_space: np.ndarray,
        observant_coordinates: np.ndarray,
        current_frame: int = 0,
    ) -> "SimpleCamera":
        """Build a camera by orbiting around a view center.

        Requires intrinsics in camera_params:
          - width, height
          - either (fx, fy) or (fovx_deg/fovy_deg)
        And orbit parameters:
          - init_azimuthm, init_elevation, init_radius
          - optional delta_a/delta_e/delta_r when move_camera=true
        """
        width = camera_params.get("width")
        height = camera_params.get("height")
        if width is None or height is None:
            raise ValueError("Procedural camera requires camera_params.width/height")

        fx = camera_params.get("fx")
        fy = camera_params.get("fy")
        if fx is None or fy is None:
            fovx_deg = camera_params.get("fovx_deg")
            fovy_deg = camera_params.get("fovy_deg")
            if fovx_deg is None and fovy_deg is None:
                raise ValueError(
                    "Procedural camera requires either fx/fy or fovx_deg/fovy_deg"
                )
            if fovx_deg is None:
                fovy = float(fovy_deg) / 180.0 * math.pi
                fy = fov2focal(fovy, float(height))
                fx = fy
            elif fovy_deg is None:
                fovx = float(fovx_deg) / 180.0 * math.pi
                fx = fov2focal(fovx, float(width))
                fy = fx
            else:
                fovx = float(fovx_deg) / 180.0 * math.pi
                fovy = float(fovy_deg) / 180.0 * math.pi
                fx = fov2focal(fovx, float(width))
                fy = fov2focal(fovy, float(height))

        init_a = camera_params.get("init_azimuthm")
        init_e = camera_params.get("init_elevation")
        init_r = camera_params.get("init_radius")
        if init_a is None or init_e is None or init_r is None:
            raise ValueError(
                "Procedural orbit camera requires init_azimuthm/init_elevation/init_radius"
            )

        if camera_params.get("move_camera", False):
            da = camera_params.get("delta_a", 0) or 0
            de = camera_params.get("delta_e", 0) or 0
            dr = camera_params.get("delta_r", 0) or 0
            az = init_a + current_frame * da
            el = init_e + current_frame * de
            rad = init_r + current_frame * dr
        else:
            az, el, rad = init_a, init_e, init_r

        position, rot = get_camera_position_and_rotation(
            az, el, rad, center_view_world_space, observant_coordinates
        )

        # Match build_camera_from_json math: treat (rot, position) as raw pose.
        tmp = np.zeros((4, 4))
        tmp[:3, :3] = rot
        tmp[:3, 3] = position
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]

        fovx = focal2fov(float(fx), float(width))
        fovy = focal2fov(float(fy), float(height))
        return SimpleCamera(
            R=R, T=T, FoVx=fovx, FoVy=fovy, width=int(width), height=int(height)
        )

    def build_camera_fixed(
        self,
        *,
        camera_params: dict,
    ) -> "SimpleCamera":
        """Build a camera from fixed extrinsics (no cameras.json).

        Expects:
          - width, height
          - either (fx, fy) or (fovx_deg/fovy_deg)
          - fixed_position: [x,y,z]
          - fixed_rotation: 3x3 matrix
        """
        width = camera_params.get("width")
        height = camera_params.get("height")
        if width is None or height is None:
            raise ValueError("Fixed procedural camera requires camera_params.width/height")

        fx = camera_params.get("fx")
        fy = camera_params.get("fy")
        if fx is None or fy is None:
            fovx_deg = camera_params.get("fovx_deg")
            fovy_deg = camera_params.get("fovy_deg")
            if fovx_deg is None and fovy_deg is None:
                raise ValueError(
                    "Fixed procedural camera requires either fx/fy or fovx_deg/fovy_deg"
                )
            if fovx_deg is None:
                fovy = float(fovy_deg) / 180.0 * math.pi
                fy = fov2focal(fovy, float(height))
                fx = fy
            elif fovy_deg is None:
                fovx = float(fovx_deg) / 180.0 * math.pi
                fx = fov2focal(fovx, float(width))
                fy = fx
            else:
                fovx = float(fovx_deg) / 180.0 * math.pi
                fovy = float(fovy_deg) / 180.0 * math.pi
                fx = fov2focal(fovx, float(width))
                fy = fov2focal(fovy, float(height))

        pos = camera_params.get("fixed_position")
        rot = camera_params.get("fixed_rotation")
        if pos is None or rot is None:
            raise ValueError("Fixed procedural camera requires fixed_position/fixed_rotation")

        position = np.array(pos, dtype=np.float32)
        rotation = np.array(rot, dtype=np.float32)
        if rotation.shape != (3, 3):
            raise ValueError(f"fixed_rotation must be 3x3, got shape {rotation.shape}")

        tmp = np.zeros((4, 4), dtype=np.float32)
        tmp[:3, :3] = rotation
        tmp[:3, 3] = position
        tmp[3, 3] = 1.0
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]

        fovx = focal2fov(float(fx), float(width))
        fovy = focal2fov(float(fy), float(height))
        return SimpleCamera(
            R=R, T=T, FoVx=fovx, FoVy=fovy, width=int(width), height=int(height)
        )
