所有 TODO 已完成。以下是本次重构的完整总结：

## 变更概览

### 新建文件

- **`physics_sim/renderer/backend_base.py`** — `RasterBackend` 抽象基类，定义统一的 `render()` 接口（输入：camera + means + cov6 + colors + opacities + bg_color，输出：`[3, H, W]` 图像 + meta dict），以及 `create_raster_backend()` 工厂函数。

- **`physics_sim/renderer/backend_diffrast.py`** — `DiffRastBackend`，将原有 `build_rasterizer()` 逻辑封装为统一接口。内部延迟导入 `diff_gaussian_rasterization`。

- **`physics_sim/renderer/backend_gsplat.py`** — `GsplatBackend`，实现 gsplat 渲染后端。包含 `_cov6_to_mat3()` 辅助函数将 (N,6) 上三角协方差转为 (N,3,3)。内部延迟导入 `gsplat`。

### 修改文件

- **`physics_sim/renderer/gs_renderer.py`**
  - `SimpleCamera` 新增 `viewmat`（row-major 4x4）和 `K`（3x3 内参矩阵）属性，同时保留原有的 `world_view_transform` / `full_proj_transform` 等属性
  - `GaussianRenderer.__init__` 接收 `raster_backend` 参数（默认 `"gsplat"`）
  - 用 `render()` 委托方法替换了原有的 `build_rasterizer()` 方法

- **`pipeline.py`**
  - 新增 `--raster_backend` CLI 参数（`gsplat` / `diffrast`）
  - 帧循环中用 `renderer.render()` 替换原来的 `rasterize()` 调用
  - 去掉了每帧的 `renderer.build_rasterizer()` 调用

- **`physics_sim/renderer/__init__.py`** — 新增导出 `RasterBackend`, `create_raster_backend`

- **`requirements.txt`** — 添加 `gsplat`

### 使用方式

```bash
# 默认使用 gsplat
python pipeline.py --ply_path ... --config ...

# 切回原有 diff_gaussian_rasterization
python pipeline.py --ply_path ... --config ... --raster_backend diffrast

# 无渲染模式（不需要任何光栅化库）
python pipeline.py --ply_path ... --config ... --no_render
```