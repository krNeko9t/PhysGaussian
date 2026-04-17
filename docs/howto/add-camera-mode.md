# How To Add a Camera Mode

1. Ensure your camera builder supports required logic.
   - default builder lives in `physics_sim/render/camera.py`
2. Register the mode resolver in `physics_sim/render/registries.py`:
   - `register_camera_mode("new_mode", resolver)`
3. Add config value in `CameraConfig.camera_mode` choices if needed.
4. Keep `run_with_rendering()` unchanged.
   - mode dispatch should happen via `resolve_camera_for_mode(...)`.

## Resolver Signature

Resolvers receive shared kwargs from the render loop:
- `camera_builder`
- `camera_params`
- `center_view_world_space`
- `observant_coordinates`
- `current_frame`
- `source_axes`
- `cameras_json` (for json-like modes)

Ignore unused kwargs via `**_`.
