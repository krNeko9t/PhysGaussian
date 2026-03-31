所有 todo 已完成。以下是实现的总结：

---

**新增文件（3个）：**

1. `physics_sim/config/material_lut.json` -- 14 种材质原型的数值参数映射表 + heaviness/stiffness/friction bin 的缩放因子，纯 JSON 可编辑
2. `physics_sim/config/geometry_lut.json` -- `geometry_form` -> `collision_geometry` 的映射（含 `use_sdf`、`alpha` 等关联参数），纯 JSON 可编辑
3. `phys_desc_to_config.py` -- 转换器主体，包含：
   - 材质加权混合逻辑（多原型 score 加权 + bin confidence 缩放）
   - `behavior_template` -> `mode` 映射
   - Static 物体的碰撞体自动配置（SVD 平面拟合参数、friction/surface 从 priors 推断、`prefer_up` 从 PLY 点云估计）
   - Backend 自动推断（全 rigid -> `newton_rigid`，全 soft -> `newton_mpm`，混合 -> `newton_vbd`，可被 `--backend-preference` 覆盖）
   - 场景级默认参数（gravity/time/camera/axis），可被 `--scene-defaults` JSON 覆盖

**修改文件（2个）：**

4. `physics_sim/config/parser.py` -- `decode_param_json` 新增第 8 个返回值 `config_backend`，从 JSON 顶层 `"backend"` 字段读取
5. `pipeline.py` -- `--backend` 默认值从 `"warp_mpm"` 改为 `"auto"`，新增 auto 解析逻辑：从 config JSON 读取 backend，未找到则 fallback 到 `warp_mpm`。旧配置文件完全向后兼容

**使用方式：**

```bash
# Step 1: 转换
python phys_desc_to_config.py \
  --phys_desc phys_desc.json \
  --ply_dir /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/projects/AnySplat/trace_output/bench_idmap_split/postprocess/gt_split \
  --output config/bench_auto_0331.json

# Step 2: 跑管线（backend 自动从 config 读取）
python pipeline.py \
  --ply_path scene_data/all.ply \
  --config config/auto_config.json \
  --output output/auto_scene \
  --camera_mode orbit --sh_degree 0 --compile_video
```