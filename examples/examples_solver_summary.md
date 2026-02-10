# Newton Examples 解算器与物理类型总结

## 一、按物理类型分类

### 1. 纯刚体 (Rigid Body Only)

| Example 名称 | 使用解算器 | 说明 |
|-------------|-----------|------|
| basic_pendulum | XPBD | 双摆 |
| basic_urdf | XPBD | URDF 刚体 |
| basic_joints | XPBD | 关节系统 |
| basic_shapes | XPBD 或 VBD | 默认 XPBD，可加 `--solver vbd` |
| basic_recording | MuJoCo | 录制 |
| example_sdf (contacts) | XPBD 或 MuJoCo | 默认 XPBD，可选 mujoco |
| robot_cartpole | MuJoCo | 机器人 |
| robot_humanoid | MuJoCo | 机器人 |
| robot_g1 | MuJoCo | 机器人 |
| robot_h1 | MuJoCo | 机器人 |
| robot_anymal_d | MuJoCo | 机器人 |
| robot_anymal_c_walk | MuJoCo | 机器人 |
| robot_policy | MuJoCo | 机器人 |
| robot_ur10 | MuJoCo | 机器人 |
| robot_panda_hydro | MuJoCo | 机器人 |
| robot_allegro_hand | MuJoCo | 机器人 |
| ik_franka / ik_h1 / ik_custom / ik_cube_stacking / ik_benchmark | MuJoCo + IK | 逆运动学 |
| selection_cartpole / selection_articulations / selection_materials / selection_multiple | MuJoCo | 选择交互 |
| sensor_contact / sensor_tiled_camera / sensor_imu | MuJoCo | 传感器 |
| cable_bend / cable_twist / cable_pile / cable_helix / cable_fixed_joints / cable_ball_joints / cable_y_junction / cable_bend_damping / cable_bundle_hysteresis | VBD | 缆绳/铰接刚体 |

### 2. 纯软体 (Soft Body / Cloth Only)

| Example 名称 | 使用解算器 | 说明 |
|-------------|-----------|------|
| cloth_bending | VBD | 布料弯曲 |
| cloth_hanging | VBD / style3d / xpbd / semi_implicit | 通过 `--solver` 选择 |
| cloth_style3d | Style3D | Style3D 布料 |
| cloth_h1 | Style3D | H1 服装 |
| cloth_twist | VBD | 布料扭转与自碰撞 |
| cloth_rolling_cloth | VBD | 卷布展开 |
| softbody_hanging | VBD | 体积软体悬挂 |

### 3. 刚体 + 软体 交互 (Rigid-Soft Interaction)

| Example 名称 | 使用解算器 | 说明 |
|-------------|-----------|------|
| cloth_franka | VBD(布料) + Featherstone(机械臂) | 机械臂与布料 |
| softbody_dropping_to_cloth | VBD | 体积软体掉到布料上 |
| falling_gift | VBD | 软体礼盒下落碰撞 |
| poker_cards_stacking | VBD | 扑克牌(薄壳) + 刚体立方体 + 球 |
| mpm_twoway_coupling | XPBD(刚体) + MPM | 刚体与 MPM 沙耦合 |
| mpm_anymal | MuJoCo + MPM | 机器人在颗粒地形 |

---

## 二、按解算器分类 (XPBD / VBD / Style3D)

### 使用 XPBD 的 Examples

- basic_pendulum
- basic_urdf
- basic_joints
- basic_shapes（默认）
- example_sdf（默认）
- mpm_twoway_coupling（刚体部分）

### 使用 VBD 的 Examples

- basic_shapes（加 `--solver vbd`）
- cloth_bending
- cloth_hanging（可选）
- cloth_twist
- cloth_franka
- cloth_rolling_cloth
- softbody_hanging
- softbody_dropping_to_cloth
- falling_gift
- poker_cards_stacking
- 全部 cable_* 示例（9 个）

### 使用 Style3D 的 Examples

- cloth_style3d
- cloth_h1
- cloth_hanging（加 `--solver style3d`）

---

## 三、实时性说明

- **XPBD、VBD、Style3D** 均为面向实时/近实时的解算器，在 GPU 上、合适网格与迭代下可做到实时。
- **能实时的典型场景**：上述用这三种解算器的 example，在默认或适中参数下一般可实时。
- **非这三种解算器**：DiffSim 系列用 SemiImplicit（偏可微），MPM 用 ImplicitMPM（算力大），Robot/IK 等用 MuJoCo（也可实时，但不在本次三种解算器范围内）。

---

## 四、速查：解算器 ↔ 物理类型

| 解算器 | 刚体 Example | 软体 Example | 刚软交互 Example |
|--------|--------------|--------------|------------------|
| XPBD | basic_pendulum, basic_urdf, basic_joints, basic_shapes, example_sdf, mpm_twoway 刚体部分 | 无 | 无 |
| VBD | basic_shapes(可选), 全部 cable_* | cloth_bending, cloth_hanging, cloth_twist, cloth_rolling_cloth, softbody_hanging | cloth_franka, softbody_dropping_to_cloth, falling_gift, poker_cards_stacking |
| Style3D | 无 | cloth_style3d, cloth_h1, cloth_hanging(可选) | 无 |
