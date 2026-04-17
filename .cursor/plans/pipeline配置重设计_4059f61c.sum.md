所有 todo 已完成。以下是实现总结：

---

**已完成的全部改动：**

### 1. `physics_sim/config/models.py` — Pydantic 配置模型
- `PlySource`, `IdMapSource` (带 discriminator)
- `ObjectTransform` (position + rotation_degrees)
- `ColliderConfig`, `FillingConfig`
- `ObjectConfig` (name, source, role, transform, initial_velocity, material, particle_filling, collider)
- `NewtonRigidConfig`, `NewtonVBDConfig`, `NewtonMPMConfig`, `NoneBackendConfig` (每种 backend 独立类型)
- `TimeConfig`, `PreprocessConfig`, `CameraConfig` (所有字段带默认值)
- `SurfaceCollider`, `BoundingBox`, `ReleaseParticlesSequentially` (边界条件)
- `SimConfig` 顶层容器

### 2. `physics_sim/presets/` — 预设工厂函数
- `materials.py` — `rubber()`, `sand()`, `jelly()`, `soft_body()`, `rigid_body()`
- `objects.py` — `wolf()`, `bread()`
- `backends.py` — `newton_rigid()`, `newton_vbd()`, `newton_mpm()`, `none_backend()`
- `cameras.py` — `orbit_camera()`, `fixed_camera()`, `json_camera()`
- `fillings.py` — `default_filling()`, `dense_filling()`

### 3. `physics_sim/backend/registry.py` + `none.py`
- `BACKEND_MATERIAL_DEFAULTS` — 每种 backend 的材质默认值
- `resolve_material()` — 用户值覆盖默认值
- `create_backend()` — 工厂函数
- `NoneBackend` — 从旧 pipeline.py 移出

### 4. `physics_sim/config/loader.py` — ~15 行 importlib 加载器
- 没有 `_ref`，没有 deep-merge，没有 ConfigLoader

### 5. `physics_sim/scene/objects.py` + `assembler.py`
- `mode` → `role` (`"dynamic"` / `"collider_only"` / `"render_only"`)
- 新增 `initial_velocity` 字段
- assembler 消费 Pydantic `ObjectConfig` (不再是 raw dict)

### 6. `physics_sim/stages/` — pipeline 拆分
- `runtime.py`, `scene_setup.py`, `backend_init.py`, `camera_setup.py`, `sim_loop.py`, `video.py`

### 7. `pipeline.py` — 瘦身为 ~80 行入口

### 8. 4 个实验 YAML → Python 配置文件
- `experiments/wolf_bread_rigid.py`, `wolf_bread_vbd.py`, `wolf_sand.py`, `bench_auto_vbd.py`

### 9. 清理
- 删除旧 `schema.py`, 旧 `loader.py`, 整个 `physics_sim/conf/` 目录
- 更新所有 `__init__.py` 引用