"""Camera models and camera-building utilities for rendering."""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import torch


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Inverse of focal2fov. fov in radians."""
    return pixels / (2.0 * math.tan(fov * 0.5))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
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
    """Minimal camera that mirrors the interface expected by rasterizers."""

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
        self.viewmat = torch.tensor(w2v, dtype=torch.float32).cuda()
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


class CameraFactory:
    """Build cameras from json and procedural camera params."""

    def build_camera_from_json(
        self,
        cameras_json_path: str,
        camera_params: dict,
        center_view_world_space=None,
        observant_coordinates=None,
        current_frame: int = 0,
        source_axes=None,
    ) -> SimpleCamera:
        from physics_sim.coord import SourceAxes as _SA, align_camera_w2c_rotation, align_camera_position

        with open(cameras_json_path) as f:
            data = json.load(f)

        default_idx = camera_params.get("default_camera_index", 0)
        show_hint = camera_params.get("show_hint", False)
        need_align = (
            source_axes is not None
            and isinstance(source_axes, _SA)
            and not source_axes.is_identity
        )

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
            raw_camera = dict(data[default_idx])
            if need_align:
                raw_camera["rotation"] = align_camera_w2c_rotation(
                    np.array(raw_camera["rotation"]), source_axes
                ).tolist()
                raw_camera["position"] = align_camera_position(
                    np.array(raw_camera["position"]), source_axes
                ).tolist()
        else:
            raw_camera = dict(data[0])
            init_a = camera_params["init_azimuth"]
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
                    init_a, init_e, init_r, center_view_world_space, observant_coordinates
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

    def build_camera_orbit(
        self,
        *,
        camera_params: dict,
        center_view_world_space: np.ndarray,
        observant_coordinates: np.ndarray,
        current_frame: int = 0,
    ) -> SimpleCamera:
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
                raise ValueError("Procedural camera requires either fx/fy or fovx_deg/fovy_deg")
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

        init_a = camera_params.get("init_azimuth")
        init_e = camera_params.get("init_elevation")
        init_r = camera_params.get("init_radius")
        if init_a is None or init_e is None or init_r is None:
            raise ValueError("Procedural orbit camera requires init_azimuth/init_elevation/init_radius")

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
        tmp = np.zeros((4, 4))
        tmp[:3, :3] = rot
        tmp[:3, 3] = position
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]
        fovx = focal2fov(float(fx), float(width))
        fovy = focal2fov(float(fy), float(height))
        return SimpleCamera(R=R, T=T, FoVx=fovx, FoVy=fovy, width=int(width), height=int(height))

    def build_camera_fixed(
        self,
        *,
        camera_params: dict,
    ) -> SimpleCamera:
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
                raise ValueError("Fixed procedural camera requires either fx/fy or fovx_deg/fovy_deg")
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
        return SimpleCamera(R=R, T=T, FoVx=fovx, FoVy=fovy, width=int(width), height=int(height))
