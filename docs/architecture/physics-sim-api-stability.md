# physics_sim 公共接口稳定与迁移说明

本说明用于约束重构后的稳定接口，降低后续内部替换成本。

## 1. 稳定公共接口（建议长期保持）

- `physics_sim.coord`
  - 保持门面导出：`SourceAxes`、`UpAxis`、`align_*`、`inverse_align_*`、`alignment_matrix*`、重力契约相关符号。
  - 内部实现位于 `physics_sim/coord/` 包（如 `axes.py`、`ops.py`、`gravity.py`、`camera.py`），调用方不应直接依赖子模块。
- `physics_sim.backend.base`
  - `PhysicsBackend` 生命周期：`initialize -> set_material -> set_boundary_conditions -> finalize -> step/get_state`。
  - 违反顺序统一抛结构化生命周期异常（`[E_LIFECYCLE]`）。
- `physics_sim.render.camera`
  - 保持 `SimpleCamera`、`CameraFactory` 对外语义。
  - 纯数学计算迁移到 `render/camera_math.py`，调用方无需改动 `camera.py` 导入路径。
- `physics_sim.preprocessing.particle_filling`
  - 保持 `fill_particles`、`get_particle_volume`、`init_filled_particles` 入口。
  - 分块执行逻辑迁移到 `preprocessing/particle_filling_chunks.py`。
- `physics_sim.sh_contract`
  - 保持 SH 张量形状、系数计数和 PLY 特征映射规则的稳定定义。

## 2. 内部可替换实现（允许继续演化）

- `physics_sim/backend/newton_common/*`
  - `boundary.py`：边界平面与摩擦映射共享逻辑。
  - `rigid_state.py`：刚体状态导出共享逻辑。
  - `rigid_geometry.py`：Rigid/VBD 共用刚体碰撞几何构建。
- `physics_sim/errors.py`
  - 统一配置错误与生命周期错误模型（含结构化错误码前缀）。

## 3. 迁移建议

- 外部调用统一指向门面模块，不直接 import 新增内部实现文件。
- 若业务代码曾捕获 `KeyError`/`ValueError` 处理 registry 未知项，迁移到捕获 `PhysicsSimConfigurationError`（仍继承 `ValueError`）。
- 若业务代码依赖 `assert` 触发前置条件错误，迁移到显式异常捕获（`PhysicsSimLifecycleError`）。

## 4. 回归检查清单

- 门面仍导出既有核心符号（`physics_sim.coord` 包、`render/camera.py`、`preprocessing/particle_filling.py`）。
- backend/render registry 未知项错误类型一致。
- 重力契约错误保持可定位上下文。
- Rigid/VBD/MPM 共享逻辑由 `newton_common` 归口，避免重复实现回流。

