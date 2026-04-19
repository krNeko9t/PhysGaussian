"""Pose and world-frame normalization helpers for cameras."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .axes import SourceAxes, alignment_matrix_np

PoseConvention = Literal["opencv_w2c", "opengl_c2w"]
WorldFrame = Literal["source", "internal"]


def normalize_pose_to_c2w(
    *,
    rotation: np.ndarray,
    position: np.ndarray,
    pose_convention: PoseConvention,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize storage-specific pose into c2w rotation/position."""
    if pose_convention == "opengl_c2w":
        # OpenGL camera frame: +Y up, -Z forward.
        # Renderer path uses an OpenCV-like camera frame: +X right, +Y down, +Z forward.
        cam_frame_fix = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        return rotation @ cam_frame_fix, position
    if pose_convention == "opencv_w2c":
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = position
        c2w = np.linalg.inv(w2c)
        return c2w[:3, :3], c2w[:3, 3]
    raise ValueError(f"Unsupported pose convention: {pose_convention}")


def align_c2w_world_frame(
    *,
    rotation_c2w: np.ndarray,
    position_c2w: np.ndarray,
    world_frame: WorldFrame,
    source_axes: SourceAxes | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map c2w pose from source/internal world into internal world."""
    if world_frame == "internal":
        return rotation_c2w, position_c2w
    if world_frame != "source":
        raise ValueError(f"Unsupported camera_world_frame: {world_frame}")
    if source_axes is None:
        raise ValueError("camera_world_frame='source' requires source_axes from scene setup")
    if source_axes.is_identity:
        return rotation_c2w, position_c2w
    align = alignment_matrix_np(source_axes)
    return align @ rotation_c2w, align @ position_c2w


def to_raw_camera_payload(
    *,
    rotation_c2w: np.ndarray,
    position_c2w: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
) -> dict:
    """Build camera payload consumed by render.camera_math."""
    return dict(
        rotation=rotation_c2w.tolist(),
        position=position_c2w.tolist(),
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
    )
