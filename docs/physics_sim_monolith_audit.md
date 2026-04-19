# physics_sim 巨文件与复杂度热点清单

本清单用于冻结本轮治理范围，避免重构中途扩散。

## 热点文件（按优先级）

1. `physics_sim/stages/sim_loop.py`
   - 典型问题：单函数跨越仿真推进、诊断、渲染输入拼装、图像写盘。
   - 主要治理对象：`run_with_rendering()`
2. `physics_sim/stages/backend_init.py`
   - 典型问题：后端生命周期编排与材质决议、碰撞平面拟合、坐标语义转换耦合。
   - 主要治理对象：`init_backend()`
3. `physics_sim/backend/newton_vbd/solver.py`
   - 典型问题：生命周期逻辑与软体网格拓扑构建混在同一文件。
   - 主要治理对象：`_create_soft_body()` 及其网格辅助逻辑
4. `physics_sim/backend/newton_rigid/solver.py`
   - 典型问题：`set_material()`、`set_boundary_conditions()` 过长且分支多。
5. `physics_sim/backend/newton_mpm/solver.py`
   - 典型问题：`initialize()`、`set_material()`、`finalize()` 仍包含较多底层细节。

## 本轮治理约束

- 稳定接口：`physics_sim/backend/base.py` 中 `PhysicsBackend` 与 `SimulationState` 保持稳定。
- 就近聚合：将材料、边界、状态导出等逻辑优先下沉到同子包邻近模块，不新增不必要中间层。
- 单向依赖：`stages -> backend API -> backend impl/common`；渲染逻辑不反向渗入后端实现。
- 复杂度门禁（告警阈值）：
  - 单文件 > 300 行
  - 单函数 > 80 行
  - 最大嵌套深度 > 4
