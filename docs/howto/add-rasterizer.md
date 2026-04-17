# How To Add a Rasterizer

1. Implement a rasterizer class with the `Rasterizer` contract:
   - input: camera/means/colors/opacities (+ cov6 or quats+scales)
   - output: `(rendered_tensor, meta_dict)`
2. Put it under `physics_sim/render/rasterizers/`.
3. Register it in `physics_sim/render/registries.py`:
   - `register_rasterizer("your_backend", factory)`
4. Use it through CLI:
   - pass `--raster_backend your_backend`
   - no core `pipeline.py` change required.

## Notes

- Keep backend imports lazy inside implementation methods if dependency is optional.
- Do not add new logic back into `physics_sim/renderer/gs_renderer.py`.
