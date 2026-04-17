# Pipeline Boundaries

## Purpose

This document defines stable boundaries for the simulation/render pipeline so
new features can be added by extension instead of core-flow rewrites.

## Layer Responsibilities

- `pipeline.py`: CLI argument parsing only.
- `physics_sim/pipeline_orchestrator.py`: high-level branch orchestration.
- `physics_sim/stages/scene_setup.py`: scene assembly + tensor packing for backend.
- `physics_sim/stages/backend_init.py`: physics backend/material/boundary setup.
- `physics_sim/stages/sim_loop.py`: runtime stepping + rendering loop logic.
- `physics_sim/render/*`: render-domain components and extension registries.

## Stable Contracts

- `SceneAssetLoader` (`physics_sim/render/interfaces.py`):
  only `load_ply(path) -> GaussianAsset`
- `RenderRuntime` (`physics_sim/render/interfaces.py`):
  camera build + SH color + rasterization runtime contract
- `GaussianAsset` (`physics_sim/render/types.py`):
  typed payload for scene assembly

## Dependency Rules

- Scene stage must not import rasterizer implementations.
- Physics backend must not import render internals.
- New code must not import `physics_sim/renderer/*` directly, except
  `physics_sim/renderer/backend_base.py` as the raster-backend factory boundary.
- `physics_sim/renderer/gs_renderer.py` has been removed; do not reintroduce
  facade-style entrypoints.

## Extension Points

- Asset loader: `register_asset_loader()` in `physics_sim/render/registries.py`
- Rasterizer backend: `register_rasterizer()` in `physics_sim/render/registries.py`
- Camera mode: `register_camera_mode()` in `physics_sim/render/registries.py`

## Migration Checklist

1. Add new module under `physics_sim/render/*`.
2. Register it in `physics_sim/render/registries.py`.
3. Keep `pipeline.py` and orchestrator untouched unless flow semantics change.
4. Add/adjust boundary tests under `tests/`.
