所有任务完成。总结一下这次重构做了什么：

---

## 完成的改动

### 核心理念
`pipeline.py` 现在完全在 **rotated world space** (旋转后的世界坐标) 操作，不再做任何归一化。每个后端在自己内部处理所需的坐标变换。

### 1. `pipeline.py` — 大幅简化
- **移除了** `transform2origin`, `shift2center111`, `transform_with_reference`, `world_to_mpm_positions`, `world_to_mpm_directions` 等归一化相关调用
- **移除了** taichi 预处理段 (`fill_particles`, `get_particle_volume`, `_compute_volumes`, `needs_volumes` 等)
- **新增** `_estimate_volumes()` — 基于粒子 bbox 的简单体积估计
- **变量重命名**: `mpm_init_pos/vol/cov` → `sim_init_pos/vol/cov`
- **边界条件**: `world` 空间的 BC 只做 `apply_rotations`（不再做 transform + shift）
- **渲染逆变换**: 只做 `apply_inverse_rotations` 和 `apply_inverse_cov_rotations`（不再做 `undoshift/undotransform/scale²`）
- **Camera json**: 局部计算 `scale_origin` 用于向后兼容的相机坐标转换

### 2. `WarpMPMBackend` — 内部自治
- `initialize()`: 接收 rotated world space 数据，内部做 `transform2origin` + `shift2center111` + 协方差缩放 + 粒子填充 + 体积计算
- `get_state()`: 自动反归一化 (`undoshift` + `undotransform` + `/scale²`)
- `set_boundary_conditions()`: 自动将 BC 点从 rotated world space 转换到 MPM space

### 3. `NewtonMPMBackend` — 动态 voxel_size
- `voxel_size` 从粒子 bbox extent 动态计算，不再依赖固定 `grid_lim`
- 移除了硬编码的 z=0 ground plane
- `bounding_box` BC 使用实际粒子 bbox 而非 `[0, grid_lim]³`

### 4. `NewtonRigidBackend` — bounding_box 修复
- `bounding_box` BC 也改为使用实际粒子 bbox

### 5. `bench_auto_0331.json` — 已确认正确
- `"backend": "newton_rigid"` 对应的 override key `"newton_rigid"` 已经匹配