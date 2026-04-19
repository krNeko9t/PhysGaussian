from __future__ import annotations

import json

import numpy as np
import pytest

from physics_sim.coord import SourceAxes, alignment_matrix_np
from physics_sim.render.camera_external import load_external_camera_raw


def _write_json(tmp_path, name: str, payload: object) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_external_camera_requires_nonnegative_index(tmp_path):
    camera_path = _write_json(
        tmp_path,
        "camera.json",
        [
            {
                "rotation": np.eye(3).tolist(),
                "position": [0.0, 0.0, 1.0],
                "width": 800,
                "height": 600,
                "fx": 500.0,
                "fy": 500.0,
            }
        ],
    )
    params = dict(
        camera_path=camera_path,
        camera_format="physgaussian",
        camera_pose_convention="opengl_c2w",
        camera_world_frame="internal",
        camera_index=-1,
    )
    with pytest.raises(ValueError, match="camera.camera_index"):
        load_external_camera_raw(camera_params=params, source_axes=SourceAxes.identity())


def test_external_camera_index_bounds(tmp_path):
    camera_path = _write_json(
        tmp_path,
        "camera.json",
        [
            {
                "rotation": np.eye(3).tolist(),
                "position": [0.0, 0.0, 1.0],
                "width": 800,
                "height": 600,
                "fx": 500.0,
                "fy": 500.0,
            }
        ],
    )
    params = dict(
        camera_path=camera_path,
        camera_format="colmap",
        camera_pose_convention="opengl_c2w",
        camera_world_frame="internal",
        camera_index=3,
    )
    with pytest.raises(IndexError, match="out of range"):
        load_external_camera_raw(camera_params=params, source_axes=SourceAxes.identity())


def test_external_camera_source_frame_alignment(tmp_path):
    camera_path = _write_json(
        tmp_path,
        "camera.json",
        [
            {
                "rotation": np.eye(3).tolist(),
                "position": [1.0, 2.0, 3.0],
                "width": 800,
                "height": 600,
                "fx": 500.0,
                "fy": 500.0,
            }
        ],
    )
    source_axes = SourceAxes.from_config(source_up="+Z", source_front="-Y")
    params = dict(
        camera_path=camera_path,
        camera_format="physgaussian",
        camera_pose_convention="opengl_c2w",
        camera_world_frame="source",
        camera_index=0,
    )
    raw = load_external_camera_raw(camera_params=params, source_axes=source_axes)
    align = alignment_matrix_np(source_axes)
    opengl_axis_fix = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    assert np.allclose(
        np.asarray(raw["rotation"], dtype=np.float32),
        align @ opengl_axis_fix,
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(raw["position"], dtype=np.float32),
        align @ np.array([1.0, 2.0, 3.0], dtype=np.float32),
        atol=1e-6,
    )


def test_external_pose_conventions_apply_expected_axis_semantics(tmp_path):
    camera_path = _write_json(
        tmp_path,
        "camera.json",
        {
            "cameras": [
                {
                    "rotation": np.eye(3).tolist(),
                    "position": [0.0, 0.0, 2.0],
                    "width": 800,
                    "height": 600,
                    "fx": 500.0,
                    "fy": 500.0,
                },
                {
                    "rotation": np.eye(3).tolist(),
                    "position": [0.0, 0.0, -2.0],
                    "width": 800,
                    "height": 600,
                    "fx": 500.0,
                    "fy": 500.0,
                },
            ]
        },
    )
    common = dict(
        camera_path=camera_path,
        camera_format="opencv",
        camera_world_frame="internal",
    )
    c2w_raw = load_external_camera_raw(
        camera_params={**common, "camera_pose_convention": "opengl_c2w", "camera_index": 0},
        source_axes=SourceAxes.identity(),
    )
    w2c_raw = load_external_camera_raw(
        camera_params={**common, "camera_pose_convention": "opencv_w2c", "camera_index": 1},
        source_axes=SourceAxes.identity(),
    )
    opengl_expected_rot = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    assert np.allclose(
        np.asarray(c2w_raw["rotation"], dtype=np.float32),
        opengl_expected_rot,
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(c2w_raw["position"], dtype=np.float32),
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(w2c_raw["rotation"], dtype=np.float32),
        np.eye(3, dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(w2c_raw["position"], dtype=np.float32),
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
        atol=1e-6,
    )
