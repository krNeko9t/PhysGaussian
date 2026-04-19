# physics_sim 基线审计（2026-04-18）

本文件用于固化 `physics_sim` 当前质量基线，作为后续治理验收对照。

## 1. 体量与复杂度基线

按行数统计，核心巨文件如下：

- `physics_sim/backend/newton_vbd/solver.py`（631）
- `physics_sim/preprocessing/particle_filling.py`（577）
- `physics_sim/backend/newton_rigid/solver.py`（496）
- `physics_sim/coord/axes.py`（164）
- `physics_sim/render/camera.py`（334）
- `physics_sim/backend/newton_mpm/solver.py`（333）

复杂度风险：

- 单文件同时包含编排、契约、算法与数据结构，导致阅读路径长、改动回归面大。
- 同类逻辑在 backend 间分叉，行为一致性难以保证。

## 2. 重复实现基线

高价值重复点：

1. 边界条件平面与摩擦解析在 `newton_rigid/solver.py`、`newton_vbd/solver.py`、`newton_mpm/boundary_conditions.py` 各自实现。
2. 刚体粒子状态导出在 `newton_rigid/state_export.py` 与 `newton_vbd/state_export.py` 存在高度相似流程。
3. 刚体碰撞几何构建在 `newton_rigid/collider_builders.py` 与 `newton_vbd/rigid_mesh.py` 双轨维护。
4. 重力 `g` 契约检查与归一化在多个 backend 材质配置路径重复。

## 3. 错误处理基线

当前并存的错误模型：

- `ValueError`（配置契约）
- `RuntimeError`（调用顺序）
- `KeyError`（部分注册表）
- `assert`（关键运行时前置条件）

问题：

- 同类问题在不同模块抛不同异常。
- 关键前置条件依赖 `assert`，在优化模式下可能失效。
- 部分日志使用 `print`，错误上下文结构不统一。

## 4. 依赖与边界基线

当前结构主干：

- 编排层：`pipeline_orchestrator.py`、`stages/*`
- 契约层：`backend/base.py`、`physics_sim/coord/gravity.py`
- 实现层：`backend/newton_*/*`、`render/*`、`preprocessing/*`

主要边界风险：

- `stages/backend_init.py` 同时承担重力解析、碰撞体平面拟合和 backend 装配。
- `render/camera.py` 同时包含投影数学、相机模型、JSON 工厂。
- `preprocessing/particle_filling.py` 混合 Taichi kernel 与高层编排。

## 5. 治理优先级（按实施顺序）

1. 统一错误契约并修复已知前置条件漏洞。
2. 收敛注册表异常类型与断言策略。
3. 抽取边界条件与刚体状态导出的共享逻辑。
4. 统一刚体几何构建路径。
5. 拆分 solver / coord / camera / particle_filling 巨文件并稳定对外接口。

