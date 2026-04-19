"""Pure math helpers for camera construction."""

from __future__ import annotations

import math

import numpy as np
import torch


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov: float, pixels: float) -> float:
    """Inverse of focal2fov. fov in radians."""
    return pixels / (2.0 * math.tan(fov * 0.5))


def getWorld2View2(
    R: np.ndarray,
    t: np.ndarray,
    translate: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scale: float = 1.0,
) -> np.ndarray:
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


def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    proj = torch.zeros(4, 4)
    z_sign = 1.0
    proj[0, 0] = 2.0 * znear / (right - left)
    proj[1, 1] = 2.0 * znear / (top - bottom)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[3, 2] = z_sign
    proj[2, 2] = z_sign * zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    return proj


def generate_camera_rotation_matrix(
    camera_to_object: np.ndarray,
    object_vertical_downward: np.ndarray,
) -> np.ndarray:
    camera_to_object = camera_to_object / np.linalg.norm(camera_to_object)
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)
    first_column = np.cross(camera_y, camera_to_object)
    return np.column_stack((first_column, camera_y, camera_to_object))


def get_point_on_sphere(
    azimuth: float,
    elevation: float,
    radius: float,
    center: np.ndarray,
    observant_coordinates: np.ndarray,
) -> np.ndarray:
    canonical = (
        np.array([
            np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
            np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
            np.sin(elevation / 180.0 * np.pi),
        ])
        * radius
    )
    return center + observant_coordinates @ canonical


def get_camera_position_and_rotation(
    azimuth: float,
    elevation: float,
    radius: float,
    view_center: np.ndarray,
    observant_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )
    rot = generate_camera_rotation_matrix(
        view_center - position, -observant_coordinates[:, 2]
    )
    return position, rot


def get_current_radius_azimuth_and_elevation(
    camera_position: np.ndarray,
    view_center: np.ndarray,
    observant_coordinates: np.ndarray,
) -> tuple[float, float, float]:
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observant_coordinates[:, 0])
    )
    if np.dot(proj_onto_hori, observant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation


def camera_extrinsics_from_raw(
    raw_camera: dict,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Convert camera dict payload to renderer-ready extrinsics/intrinsics."""
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = raw_camera["rotation"]
    tmp[:3, 3] = raw_camera["position"]
    tmp[3, 3] = 1
    c2w = np.linalg.inv(tmp)
    rot = c2w[:3, :3].transpose()
    trans = c2w[:3, 3]
    width = int(raw_camera["width"])
    height = int(raw_camera["height"])
    fovx = focal2fov(raw_camera["fx"], width)
    fovy = focal2fov(raw_camera["fy"], height)
    return rot, trans, width, height, fovx, fovy

