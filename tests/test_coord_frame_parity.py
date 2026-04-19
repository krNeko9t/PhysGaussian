from __future__ import annotations

import numpy as np

from physics_sim.coord import (
    SourceAxes,
    align_c2w_world_frame,
    alignment_matrix_np,
    normalize_pose_to_c2w,
    to_raw_camera_payload,
)


def test_normalize_pose_to_c2w_opencv_matches_inverse_w2c():
    rotation_w2c = np.eye(3, dtype=np.float32)
    position_w2c = np.array([0.0, 0.0, -2.0], dtype=np.float32)
    rot_c2w, pos_c2w = normalize_pose_to_c2w(
        rotation=rotation_w2c,
        position=position_w2c,
        pose_convention="opencv_w2c",
    )
    assert np.allclose(rot_c2w, np.eye(3, dtype=np.float32), atol=1e-6)
    assert np.allclose(pos_c2w, np.array([0.0, 0.0, 2.0], dtype=np.float32), atol=1e-6)


def test_normalize_pose_to_c2w_opengl_applies_camera_axis_fix():
    rotation_c2w = np.eye(3, dtype=np.float32)
    position_c2w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rot_norm, pos_norm = normalize_pose_to_c2w(
        rotation=rotation_c2w,
        position=position_c2w,
        pose_convention="opengl_c2w",
    )
    expected_rot = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    assert np.allclose(rot_norm, expected_rot, atol=1e-6)
    assert np.allclose(pos_norm, position_c2w, atol=1e-6)


def test_align_c2w_world_frame_matches_source_alignment():
    source_axes = SourceAxes.from_config(source_up="+Z", source_front="-Y")
    align = alignment_matrix_np(source_axes)
    rot_src = np.eye(3, dtype=np.float32)
    pos_src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rot_internal, pos_internal = align_c2w_world_frame(
        rotation_c2w=rot_src,
        position_c2w=pos_src,
        world_frame="source",
        source_axes=source_axes,
    )
    assert np.allclose(rot_internal, align @ rot_src, atol=1e-6)
    assert np.allclose(pos_internal, align @ pos_src, atol=1e-6)


def test_to_raw_camera_payload_matches_camera_math_contract():
    raw = to_raw_camera_payload(
        rotation_c2w=np.eye(3, dtype=np.float32),
        position_c2w=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        width=1920,
        height=1080,
        fx=1000.0,
        fy=1001.0,
    )
    assert raw["width"] == 1920
    assert raw["height"] == 1080
    assert isinstance(raw["fx"], float)
    assert isinstance(raw["fy"], float)
    assert np.allclose(np.asarray(raw["rotation"], dtype=np.float32), np.eye(3), atol=1e-6)
