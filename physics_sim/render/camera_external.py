"""External camera parsing and coordinate normalization helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from physics_sim.coord import (
    SourceAxes,
    align_c2w_world_frame,
    normalize_pose_to_c2w,
    to_raw_camera_payload,
)

_SUPPORTED_FORMATS = {"colmap", "blender", "nerfstudio", "physgaussian", "opencv", "opengl"}
_SUPPORTED_CONVENTIONS = {"opencv_w2c", "opengl_c2w"}
_SUPPORTED_WORLD_FRAMES = {"source", "internal"}


def load_external_camera_raw(
    *,
    camera_params: dict,
    source_axes: SourceAxes | None,
) -> dict:
    """Load one external camera and normalize it to internal c2w payload."""
    camera_path = _require_string(camera_params, "camera_path")
    camera_format = _require_string(camera_params, "camera_format")
    pose_convention = _require_string(camera_params, "camera_pose_convention")
    world_frame = _require_string(camera_params, "camera_world_frame")
    camera_index = _require_nonnegative_int(camera_params, "camera_index")

    if camera_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported camera_format='{camera_format}'. "
            f"Supported: {sorted(_SUPPORTED_FORMATS)}"
        )
    if pose_convention not in _SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unsupported camera_pose_convention='{pose_convention}'. "
            f"Supported: {sorted(_SUPPORTED_CONVENTIONS)}"
        )
    if world_frame not in _SUPPORTED_WORLD_FRAMES:
        raise ValueError(
            f"Unsupported camera_world_frame='{world_frame}'. "
            f"Supported: {sorted(_SUPPORTED_WORLD_FRAMES)}"
        )

    payload = _load_json_payload(camera_path)
    if camera_format == "nerfstudio":
        rotation, position, width, height, fx, fy = _parse_nerfstudio_payload(
            payload, camera_index, camera_path
        )
    else:
        rotation, position, width, height, fx, fy = _parse_standard_payload(
            payload, camera_index, camera_path
        )

    rot_c2w, pos_c2w = normalize_pose_to_c2w(
        rotation=rotation,
        position=position,
        pose_convention=pose_convention,
    )
    rot_c2w, pos_c2w = align_c2w_world_frame(
        rotation_c2w=rot_c2w,
        position_c2w=pos_c2w,
        world_frame=world_frame,
        source_axes=source_axes,
    )
    _validate_rotation(rot_c2w, camera_path)
    _validate_intrinsics(width, height, fx, fy, camera_path)
    return to_raw_camera_payload(
        rotation_c2w=rot_c2w,
        position_c2w=pos_c2w,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
    )


def _load_json_payload(camera_path: str) -> Any:
    with open(camera_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_standard_payload(
    payload: Any,
    camera_index: int,
    camera_path: str,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    entry = _select_camera_entry(payload, camera_index, camera_path)
    if not isinstance(entry, dict):
        raise ValueError(f"Camera entry at index={camera_index} must be an object")
    rotation, position = _parse_pose_from_entry(entry, camera_path)
    width, height, fx, fy = _parse_intrinsics(entry, camera_path)
    return rotation, position, width, height, fx, fy


def _parse_nerfstudio_payload(
    payload: Any,
    camera_index: int,
    camera_path: str,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    if not isinstance(payload, dict):
        raise ValueError("Nerfstudio camera file must be a JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Nerfstudio camera file requires a 'frames' array")
    if camera_index >= len(frames):
        raise IndexError(
            f"camera_index={camera_index} out of range for {camera_path} "
            f"(frames={len(frames)})"
        )
    frame = frames[camera_index]
    if not isinstance(frame, dict):
        raise ValueError("Each nerfstudio frame must be an object")
    if "transform_matrix" not in frame:
        raise ValueError("Nerfstudio frame must contain 'transform_matrix'")

    matrix = _as_matrix(frame["transform_matrix"], shape=(4, 4), name="transform_matrix")
    rotation = matrix[:3, :3]
    position = matrix[:3, 3]
    width, height, fx, fy = _parse_intrinsics(frame, camera_path, fallback=payload)
    return rotation, position, width, height, fx, fy


def _select_camera_entry(payload: Any, camera_index: int, camera_path: str) -> Any:
    entries = _extract_entries(payload, camera_path)
    if camera_index >= len(entries):
        raise IndexError(
            f"camera_index={camera_index} out of range for {camera_path} "
            f"(entries={len(entries)})"
        )
    return entries[camera_index]


def _extract_entries(payload: Any, camera_path: str) -> list:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported camera payload type: {type(payload).__name__}")
    for key in ("cameras", "frames"):
        entries = payload.get(key)
        if isinstance(entries, list):
            return entries
    if "rotation" in payload or "transform_matrix" in payload:
        return [payload]
    raise ValueError(
        f"Cannot find camera entries in {camera_path}. "
        "Expected list, or dict with 'cameras'/'frames'."
    )


def _parse_pose_from_entry(entry: dict, camera_path: str) -> tuple[np.ndarray, np.ndarray]:
    if "rotation" in entry and "position" in entry:
        rotation = _as_matrix(entry["rotation"], shape=(3, 3), name="rotation")
        position = _as_vector(entry["position"], size=3, name="position")
        return rotation, position
    if "transform_matrix" in entry:
        matrix = _as_matrix(entry["transform_matrix"], shape=(4, 4), name="transform_matrix")
        return matrix[:3, :3], matrix[:3, 3]
    if "qvec" in entry and "tvec" in entry:
        qvec = _as_vector(entry["qvec"], size=4, name="qvec")
        tvec = _as_vector(entry["tvec"], size=3, name="tvec")
        return _quaternion_to_rotation(qvec), tvec
    raise ValueError(
        f"Camera entry in {camera_path} must provide "
        "'rotation+position', 'transform_matrix', or 'qvec+tvec'."
    )


def _parse_intrinsics(
    entry: dict,
    camera_path: str,
    fallback: dict | None = None,
) -> tuple[int, int, float, float]:
    width = _get_optional(entry, ["width", "w"], fallback=fallback)
    height = _get_optional(entry, ["height", "h"], fallback=fallback)
    fx = _get_optional(entry, ["fx", "fl_x"], fallback=fallback)
    fy = _get_optional(entry, ["fy", "fl_y"], fallback=fallback)
    if fy is None and fx is not None:
        fy = fx
    if fx is None and fy is not None:
        fx = fy
    if width is None or height is None or fx is None or fy is None:
        raise ValueError(
            f"Camera intrinsics missing in {camera_path}. "
            "Need width/height/fx/fy (or w/h/fl_x/fl_y)."
        )
    return int(width), int(height), float(fx), float(fy)


def _validate_rotation(rotation: np.ndarray, camera_path: str) -> None:
    if rotation.shape != (3, 3):
        raise ValueError(f"{camera_path}: rotation must be 3x3, got {rotation.shape}")
    should_be_identity = rotation @ rotation.T
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-4):
        raise ValueError(f"{camera_path}: rotation is not orthonormal")
    det = float(np.linalg.det(rotation))
    if not np.isclose(det, 1.0, atol=1e-4):
        raise ValueError(f"{camera_path}: det(rotation) must be +1, got {det}")


def _validate_intrinsics(
    width: int,
    height: int,
    fx: float,
    fy: float,
    camera_path: str,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"{camera_path}: width/height must be positive")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"{camera_path}: fx/fy must be positive")


def _require_string(camera_params: dict, key: str) -> str:
    value = camera_params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"camera.{key} must be a non-empty string")
    return value.strip()


def _require_nonnegative_int(camera_params: dict, key: str) -> int:
    value = camera_params.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"camera.{key} must be a non-negative integer")
    return value


def _as_matrix(value: Any, *, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _as_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr


def _get_optional(entry: dict, keys: list[str], fallback: dict | None = None) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    if fallback is None:
        return None
    for key in keys:
        if key in fallback:
            return fallback[key]
    return None


def _quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    norm = np.linalg.norm(qvec)
    if norm < 1e-8:
        raise ValueError("qvec cannot be near zero")
    qw, qx, qy, qz = (qvec / norm).tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
