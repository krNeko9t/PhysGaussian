# SH 系数契约

本文档约束球谐系数（SH）在加载、场景组装、仿真和渲染各阶段的统一语义，避免跨层隐式耦合。

## 1. 张量形状

- 统一 SH 张量形状为 `(N, C, 3)`，其中 `C = (sh_degree + 1)^2`。
- PLY 输入中：
  - `f_dc_*` 对应首个系数（`C` 的第 0 项）；
  - `f_rest_*` 的期望数量为 `3 * C - 3`。
- 预处理若需展平，使用 `(N, C*3)`，恢复时必须回到 `(N, C, 3)`。

## 2. 坐标系与方向变换

- SH 系数存储在 PLY 原始坐标系。
- 仿真状态中的位置在内部 Y-up 坐标系。
- 视线方向计算流程固定为：
  1. `view_dir = position_internal - camera_center_internal`
  2. `view_dir = view_rotations @ view_dir`（可选；必须与粒子数量一一对应）
  3. `view_dir = alignment_inv * view_dir`（可选；将方向从内部坐标系映射回 PLY 坐标系）
  4. 归一化后送入 `eval_sh`

## 3. 模块边界

- `physics_sim/sh_contract.py`：SH 形状与计数规则（稳定契约）
- `physics_sim/render/sh_eval.py`：SH 数学求值（可替换实现）
- `physics_sim/render/gaussian_geometry.py`：协方差相关几何计算（可替换实现）
- `physics_sim/render/sh_colorizer.py`：方向计算 + SH 到 RGB 转换（稳定入口）

## 4. 不变量检查

- `convert_sh` 的 `view_rotations` 若提供，形状必须严格为 `(N, 3, 3)`。
- `alignment_inv` 若提供，形状必须严格为 `(3, 3)`。
- `shs` 输入必须与 `sh_degree` 匹配：`(N, C, 3)`。

