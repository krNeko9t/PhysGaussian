# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Conda env: `ls2_warp` — activate before running anything (`conda activate ls2_warp`).
- Heavy runtime deps: `torch`, `warp` / `warp_lang`, `newton`, `gsplat`. GPU (CUDA) required for real runs; tests that need the full stack skip themselves via `pytest.importorskip` on CPU-only boxes.
- Repo root is placed on `sys.path` by `conftest.py`, so tests and scripts can `import physics_sim` directly.

## Common commands

Run a simulation (headless or with rendering):
```
python pipeline.py --config experiments/wolf_bread_rigid.py
python pipeline.py --config experiments/wolf_bread_rigid.py --no_render          # physics only
python pipeline.py --config experiments/wolf_bread_rigid.py --compile_video      # stitch frames
python pipeline.py --config experiments/wolf_sand.py --raster_backend diffrast   # alternative rasterizer
python pipeline.py --config experiments/xxx.py --sh_degree 3                     # SH degree (0..4)
```
Experiment configs are Python files that must define a module-level `config: SimConfig` (see `physics_sim/config/loader.py`).

Tests:
```
pytest                                          # whole suite
pytest tests/test_scene_config.py               # single file
pytest tests/test_vbd_apply_constraints.py -k pin_to_world
```
Several test modules (VBD/MPM apply_constraints, etc.) only run on a GPU box with torch+warp+newton available.

Complexity guardrails (CI-style linter for file/function size & nesting):
```
python -m physics_sim.complexity_guardrails --root physics_sim
python -m physics_sim.complexity_guardrails --root physics_sim --strict
```
Limits: 300 lines/file, 80 lines/function, nesting depth 4 (`physics_sim/complexity_guardrails.py`).

Config migration (legacy JSON → scene-graph Python):
```
python scripts/migrate_configs_to_scene.py
```

## Architecture (big picture)

### Entry flow
`pipeline.py` is CLI-only. It builds a `PipelineRequest` and hands off to `physics_sim/pipeline_orchestrator.py::PipelineOrchestrator`, which runs stages in this fixed order (`physics_sim/stages/`):
1. `runtime.init_runtime(backend_type)` — warp/newton runtime bring-up.
2. `scene_setup.setup_scene(...)` — parse scene, load PLYs, align to internal Y-up, concatenate tensors, resolve constraints.
3. `backend_init.init_backend(...)` — construct `PhysicsBackend`, feed material + boundary conditions, `finalize()`, then `apply_constraints()`.
4. Branch on `request.no_render`:
   - headless → `sim_loop.run_headless(...)`
   - render → `camera_setup.setup_camera(...)` + `sim_loop.run_with_rendering(...)` (+ optional `video.compile_video`).

The boundaries between these stages are treated as a contract — see `docs/architecture/pipeline-boundaries.md`. In particular: **scene stage must not import rasterizer implementations; physics backends must not import render internals;** do not reintroduce façade modules like the old `renderer/gs_renderer.py`.

### Config layer
Everything is typed Pydantic (`physics_sim/config/models.py`, `physics_sim/config/scene.py`). There is no YAML/JSON experiment loader — experiments construct `SimConfig` in Python.
- `SimConfig.backend` is a discriminated union (`newton_rigid | newton_vbd | newton_mpm | none`); backends read their own config directly from this object at construction — solver numerical options do **not** flow through `set_material`/`initialize`.
- `SimConfig.scene: SceneConfig` holds `parts` (`PartConfig`) and `constraints` (`PinToWorld | PinToBody | CollideOnly`). This replaced the older flat `objects` list; a part with `material=None` is render-only, a part referenced by `CollideOnly` is collider-only, everything else is dynamic.
- `MaterialSpec` is a discriminated union (`RigidMaterial | MPMMaterial | VBDMaterial`), with `MPMMaterial.jelly()/sand()/snow()/...` factories for preset overrides.
- Legacy JSON configs still exist under `config/legacy/` but are not loaded by `pipeline.py`; `scripts/migrate_configs_to_scene.py` ports them to the new Python scene-graph form.

### Backend contract (`physics_sim/backend/base.py`)
All physics backends implement `PhysicsBackend` with a strict lifecycle:

```
initialize(pos, vol, cov, [init_quats, init_scales])
  → set_material(MaterialSetupSpec)
  → set_boundary_conditions(bcs, time)
  → finalize()
  → apply_constraints(list[ResolvedConstraint])   # default raises on non-empty
  → (per-frame) pre_step(dt, f) → step(dt, f) → get_state()
```
Order violations raise `PhysicsSimLifecycleError` (`[E_LIFECYCLE]` prefix) from `physics_sim/errors.py`. `create_backend(cfg, device)` in `physics_sim/backend/registry.py` is the only entry point — do not instantiate backend classes directly from higher layers. Cross-backend helpers live in `physics_sim/backend/newton_common/` (boundary mapping, rigid state export, rigid collision geometry, pin kernel) — add to that package rather than duplicating across `newton_rigid/`, `newton_vbd/`, `newton_mpm/`.

### Scene graph → backend bridge
Scene-config constraints (`PinToWorld`/`PinToBody`/`CollideOnly`) are declarative. `physics_sim/scene/constraint_resolver.py::resolve_constraints` turns them into concrete `Resolved*` dataclasses with global particle indices, then `scene_setup` threads them through to `backend.apply_constraints(...)`. Each backend translates them into its own primitives (e.g. particle mass zeroing, `setup_collider`, per-step kernels invoked from `pre_step`). `PartRuntimeInfo` (in `physics_sim/backend/spec.py`) is the single source of truth for per-part index ranges.

### Coordinate system
Source PLY data can be any axis convention; `PreprocessConfig.source_up`/`source_front` declare it. `physics_sim/coord/` is the public facade — import `SourceAxes`, `align_positions`, `align_covariances`, `align_quats`, etc. from `physics_sim.coord`, never from its sub-modules. Everything downstream of `scene_setup` runs in **internal Y-up**. For SH evaluation, `alignment_inv` converts view directions back to source axes (see `docs/architecture/sh-coefficient-contract.md`).

### SH coefficients (`physics_sim/sh_contract.py`)
SH tensors are canonically shaped `(N, C, 3)` with `C = (sh_degree+1)**2`. `_MAX_SH_DEGREE = 4`. PLY `f_dc_*` maps to coefficient 0; `f_rest_*` must have `3*C - 3` features. Preprocessing may flatten to `(N, C*3)` but must restore `(N, C, 3)` before handing to the renderer. Any new SH-touching code must round-trip through `sh_contract` helpers, not hand-rolled reshapes.

### Render layer
`physics_sim/render/runtime.py::GaussianRenderRuntime` composes three pieces (camera, SH colorizer, rasterizer) via registries in `physics_sim/render/registries.py`. To extend:
- New rasterizer → class under `physics_sim/render/rasterizers/` + `register_rasterizer("name", factory)`; select via `--raster_backend name`. See `docs/howto/add-rasterizer.md`.
- New camera format (most extensions) → add to `CameraConfig.camera_format`, extend the parser in `physics_sim/render/camera_external.py`. See `docs/howto/add-camera-mode.md`.
- New asset loader → `register_asset_loader(...)`.
Keep heavy/optional imports lazy inside implementations. Do not import render internals from physics or scene code.

## Conventions

- From `.cursor/rules/coding-standards.mdc`: keep local complexity low and code self-documenting; avoid god-files and god-functions. The complexity guardrails above are enforced mechanically.
- Errors: use `configuration_error` / `lifecycle_error` / `unknown_registry_error` factories from `physics_sim/errors.py` so messages carry the `[E_CONFIG]` / `[E_LIFECYCLE]` prefix and structured context.
- Historical refactor plans and post-mortems live under `.cursor/plans/`; consult them before reopening settled design decisions (pipeline boundary split, coordinate v2, SH decoupling, scene-graph introduction, etc.).
