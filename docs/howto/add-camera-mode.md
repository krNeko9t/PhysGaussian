# How To Add a Camera Mode

## Built-in Modes

- `orbit`: procedural orbit camera in internal Y-up frame.
- `fixed`: procedural fixed camera in internal Y-up frame.
- `external`: load camera pose/intrinsics from file and normalize to internal Y-up.

Most extensions should be implemented as a new `camera_format` inside
`external`, instead of introducing a new `camera_mode`.

## External Camera Contract

When `camera_mode="external"`, config must provide:

- `camera_path`: path to the camera file.
- `camera_format`: one of `colmap|blender|nerfstudio|physgaussian|opencv|opengl`.
- `camera_pose_convention`: `opencv_w2c` or `opengl_c2w`.
- `camera_world_frame`: `source` or `internal`.
- `camera_index`: non-negative index in the camera payload.

The parser normalizes all formats to internal `c2w` and then builds `SimpleCamera`.

## Add a New Camera Format

1. Add format name to `CameraConfig.camera_format` in `physics_sim/config/models.py`.
2. Implement parser branch in `physics_sim/render/camera_external.py`.
3. Keep `run_with_rendering()` unchanged.
   - dispatch remains in `resolve_camera_for_mode(...)`.
4. Add/extend tests in `tests/test_camera_external_contract.py`.

## Resolver Signature

Resolvers receive shared kwargs from the render loop:
- `camera_builder`
- `camera_params`
- `center_view_world_space`
- `observant_coordinates`
- `current_frame`
- `source_axes`

Ignore unused kwargs via `**_`.
