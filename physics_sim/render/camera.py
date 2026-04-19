"""Camera models and camera-building utilities for rendering."""

from __future__ import annotations

import json

import numpy as np
import torch

from physics_sim.render.camera_math import (
    camera_extrinsics_from_raw,
    focal2fov,
    fov2focal,
    getProjectionMatrix,
    getWorld2View2,
    get_camera_position_and_rotation,
    get_current_radius_azimuth_and_elevation,
)


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
        from physics_sim.coord import (
            SourceAxes as _SA,
            align_camera_position,
            align_camera_w2c_rotation,
            alignment_matrix_np,
        )

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
            raise RuntimeError(
                "camera.show_hint requested. "
                f"default_camera_index={default_idx}, azimuth={a}, elevation={e}, radius={r}. "
                "Please update camera parameters and rerun."
            )

        if default_idx > -1:
            raw_camera = dict(data[default_idx])
            if need_align:
                A_np = alignment_matrix_np(source_axes)
                raw_camera["rotation"] = align_camera_w2c_rotation(
                    np.array(raw_camera["rotation"]), A_np
                ).tolist()
                raw_camera["position"] = align_camera_position(
                    np.array(raw_camera["position"]), A_np
                ).tolist()
        else:
            raw_camera = dict(data[0])
            init_a = camera_params["init_azimuth"]
            init_e = camera_params["init_elevation"]
            init_r = camera_params["init_radius"]
            if init_a is None or init_e is None or init_r is None:
                raise ValueError(
                    "camera mode=json with default_camera_index<0 requires "
                    "init_azimuth/init_elevation/init_radius"
                )
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

        R, T, width, height, fovx, fovy = camera_extrinsics_from_raw(raw_camera)
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
                fovy = float(fovy_deg) / 180.0 * np.pi
                fy = fov2focal(fovy, float(height))
                fx = fy
            elif fovy_deg is None:
                fovx = float(fovx_deg) / 180.0 * np.pi
                fx = fov2focal(fovx, float(width))
                fy = fx
            else:
                fovx = float(fovx_deg) / 180.0 * np.pi
                fovy = float(fovy_deg) / 180.0 * np.pi
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
        raw_camera = dict(
            rotation=rot,
            position=position,
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
        )
        R, T, _, _, _, _ = camera_extrinsics_from_raw(raw_camera)
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
                fovy = float(fovy_deg) / 180.0 * np.pi
                fy = fov2focal(fovy, float(height))
                fx = fy
            elif fovy_deg is None:
                fovx = float(fovx_deg) / 180.0 * np.pi
                fx = fov2focal(fovx, float(width))
                fy = fx
            else:
                fovx = float(fovx_deg) / 180.0 * np.pi
                fovy = float(fovy_deg) / 180.0 * np.pi
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

        raw_camera = dict(
            rotation=rotation,
            position=position,
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
        )
        R, T, _, _, _, _ = camera_extrinsics_from_raw(raw_camera)
        fovx = focal2fov(float(fx), float(width))
        fovy = focal2fov(float(fy), float(height))
        return SimpleCamera(R=R, T=T, FoVx=fovx, FoVy=fovy, width=int(width), height=int(height))
