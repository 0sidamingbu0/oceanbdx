# OceanBDX 整体方案架构

复刻迪士尼BDX的双足机器人控制系统, 部署于 Jetson Orin Nano, 训练采用 IsaacLab。

## 1. 系统总览

```
┌─────────────────────────── IsaacLab (训练) ───────────────────────────┐
│  ocean URDF→USD → velocity任务训练 (rsl_rl) → 导出 policy.onnx        │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │ policy.onnx
              ┌──────────────────┴───────────────────┐
              ▼                                      ▼
┌──────── sim2sim (MuJoCo) ────────┐   ┌──── sim2real (Jetson Orin Nano) ────┐
│ sim2sim/mujoco_sim.py            │   │ oceanbdx_run (C++)                  │
│  - 同一套观测/动作/FSM逻辑       │   │  ┌─────────── 线程结构 ───────────┐ │
│  - 验证policy与配置正确性        │   │  │ 左腿485轮询线程  (~350Hz)      │ │
└──────────────────────────────────┘   │  │ 右腿485轮询线程  (~350Hz)      │ │
                                       │  │ IMU读取线程     (500Hz+)       │ │
                                       │  │ 主控制循环      (200Hz)        │ │
                                       │  │ 键盘/手柄线程   (50Hz)         │ │
                                       │  └────────────────────────────────┘ │
                                       └─────────────────────────────────────┘
```

## 2. 硬件拓扑

| 设备 | 接口 | 设备名(udev) | 说明 |
|------|------|--------------|------|
| 右腿 5×GO-M8010-6 | USB转485 #1 | `/dev/ttyright` | 电机ID 1-5 |
| 左腿 5×GO-M8010-6 | USB转485 #2 | `/dev/ttyleft` | 电机ID 1-5 (脖子n1=ID6同总线, 暂不启用) |
| YIS320 IMU | USB转串口 | `/dev/ttyimu` | 460800bps, yesense协议 |
| 脖子 3×飞特舵机 | USB转485 #3 | `/dev/ttyneck` | SMS_STS协议, **暂只移植驱动不控制** |
| USB手柄 | USB 2.4G无线 | `/dev/input/js0` | Linux joystick, XInput (如罗技F710, 可选, 也可键盘) |
| 电池 BMS | USB转串口 | `/dev/ttybat` | 9600bps, A5协议 (可选监控) |

udev固定命名规则见 `config/udev/99-oceanbdx.rules`。

## 3. 软件分层

```
oceanbdx/
├── drivers/                  # 从 sarocean 提取的原始驱动 (尽量不改动)
│   ├── unitree_motor/        # 宇树串口电机SDK (预编译.so, 含Arm64版)
│   ├── feetech/              # 飞特SCServo舵机驱动
│   └── yis_imu/              # YIS320 yesense协议解析 (C)
├── include/oceanbdx/ + src/  # 本工程核心代码
│   ├── leg_driver            # 单腿485总线驱动 (线程+无锁缓存, 输出轴单位)
│   ├── imu_driver            # IMU线程 (双缓冲无锁发布)
│   ├── neck_driver           # 脖子驱动封装 (暂不启用)
│   ├── gamepad_driver        # USB手柄 (Linux joystick, 双缓冲无锁发布)
│   ├── battery_driver        # 电池BMS A5串口协议 (线程收发, 双缓冲发布)
│   ├── calibration           # 电机零位↔URDF零位换算 + 上电坐姿校验
│   ├── policy                # ONNX策略推理 (观测构造与IsaacLab对齐)
│   ├── fsm                   # 状态机
│   └── main                  # 主控制循环 200Hz
├── tests/                    # 分步调试程序 (见第6节)
├── config/oceanbdx.yaml      # 全部参数: 端口/零位/增益/策略缩放
├── description/              # ocean URDF + mesh (从sarocean拷贝, mesh已转二进制并简化)
├── sim2sim/                  # MuJoCo仿真 (ocean_scene.xml + mujoco_sim.py)
└── scripts/                  # urdf2mjcf.py, measure_offset.py
```

### 关节向量约定

全部模块统一使用URDF顺序与URDF坐标 (站立=零位):

```
[leg_r1, leg_r2, leg_r3, leg_r4, leg_r5, leg_l1, leg_l2, leg_l3, leg_l4, leg_l5]
```

★ IsaacLab 转USD后关节顺序可能变化(常见BFS交错), 训练后必须核对
`robot.joint_names`, 不一致时修改 `config/oceanbdx.yaml` 的顺序与映射。

### 单位换算 (M8010, 减速比6.33)

宇树SDK的 q/dq/kp/kd/tau 均为**转子侧**, `leg_driver` 内统一换算为输出轴:

```
q_rotor = q_out × 6.33        tau_out = tau_rotor × 6.33
kp_rotor = kp_out / 6.33²     kd_rotor = kd_out / 6.33²
```

## 4. 零位与标定方案

### 4.1 零位定义

- **URDF零位 = 站立姿态** (训练、仿真、部署统一)
- **电机不支持零位校正**, 通过测量限位位置的电机角度值和URDF角度值完成标定:
  - `q_motor_offset`: 限位位置的电机输出轴角度 (实测)
  - `urdf_offset`: 限位位置在URDF坐标系下的角度 (由可视化工具测量)
- 换算: `q_urdf = direction × (q_motor - q_motor_offset) + urdf_offset`

### 4.2 标定测量流程

**q_motor_offset** (电机限位读数):
1. 将每个关节缓慢顶到结构限位
2. `./test_calibration config/oceanbdx.yaml limit` → 按回车抓拍
3. 将输出填入 `calibration.q_motor_offset`

**urdf_offset** (URDF限位角度):
1. `python3 scripts/urdf2mjcf.py` 生成可视化模型
2. `python3 scripts/measure_offset.py` 打开MuJoCo viewer
3. 对照实物把模型每个关节拖到结构限位位置
4. 终端读出各关节角度 → 填入 `calibration.urdf_offset`
5. 同样方法摆出底座坐姿 → 填入 `calibration.sit_pose`

### 4.3 上电流程 (解决单圈绝对值+减速机的多圈歧义)

M8010转子单圈绝对值编码器经6.33减速后, 输出轴每 360°/6.33≈56.9° 读数重复。
因此:

1. 机器人放在**专用底座上坐姿上电**, 各关节由底座约束在已知小角度范围内
2. 状态机 `BOOT_CHECK`: 读取全部关节, 校验 `|q_urdf - sit_pose| < boot_tolerance(0.3rad)`
3. 校验通过才允许使能; 失败则停在PASSIVE并提示越界关节
   (说明上电时关节不在预期圈数内, 需手动摆正后重新上电)

`boot_tolerance` 必须 < 56.9°/2 ≈ 0.49 rad, 默认0.30 rad留出余量。

## 5. 状态机设计 (启动→站立→RL)

```
PASSIVE ──0──▶ BOOT_CHECK ──通过──▶ SIT_HOLD ──1──▶ STAND_UP ──完成──▶ RL_BALANCE ──2──▶ RL_WALK
   ▲                │失败                            (脚本插值)            (cmd=0)    ◀──3──┘
   └────────────────┘                  任意状态 ──9/姿态保护──▶ DAMPING (kd阻尼软停)
```

**起立用脚本而不是RL** (设计决策):

- 起立是确定性大范围姿态变化, 余弦插值脚本 (3s, sit_pose→stand_pose) 最安全可控;
- RL策略只在站立附近的状态分布内训练 (自平衡+行走), 起立完成瞬间切入RL,
  此时机器人姿态≈训练初始分布, 衔接风险最小;
- 若以后想做RL起立, 在FSM新增状态复用同一policy接口即可。

**RL_BALANCE 与 RL_WALK 共用同一policy**, 仅速度指令不同 (0 vs 手柄/固定速度),
这与IsaacLab velocity任务的训练方式一致 (指令包含0速度的采样范围)。

**安全保护**:
- 姿态保护: projected_gravity_z > -0.5 (倾倒约60°) → DAMPING
- 软限位: RL输出目标角clamp到URDF限位内缩0.02rad
- 力矩限幅: 逐关节 torque_limits
- Ctrl+C / 退出: 自动发阻尼命令再断电机

## 6. 分步调试路线 (bring-up)

按顺序执行, 每步通过后再进行下一步:

| 步骤 | 程序 | 验证内容 | 通过标准 |
|------|------|----------|---------|
| 1 | `test_leg_motor /dev/ttyright 5` | 单腿只读 | 5电机在线, >200Hz, 手转关节读数正确 |
| 2 | `test_leg_motor /dev/ttyright 5 hold` | 小增益保持 | 关节有弹性阻力, 无抖动 |
| 3 | 同上换 `/dev/ttyleft` | 另一条腿 | 同上 |
| 4 | `test_imu` | IMU | 频率正常, 静止时quat≈(1,0,0,0), accel_z≈9.8 |
| 5 | `test_neck` | 飞特舵机驱动 | 3舵机可读位置 (只验证驱动, 不控制) |
| 6 | `test_gamepad` | USB手柄 | connected, 摇杆/按键数据正确 |
| 6b | `test_battery` | 电池BMS | VALID, 电压/SOC/电流在合理范围 |
| 7 | `measure_offset.py` + `test_calibration limit` | 测urdf_offset/q_motor_offset/sit_pose | 填好config标定段 |
| 8 | `test_calibration` | 零位换算+方向 | 坐姿boot check PASS; 转动方向与URDF一致(否则改direction) |
| 9 | `mujoco_sim.py --no-policy` | 起立脚本 | 仿真中能从坐姿站起不倒 (需真实sit_pose) |
| 10 | IsaacLab训练 → `mujoco_sim.py` | sim2sim | 仿真中RL站立稳定, 能定速行走 |
| 11 | `oceanbdx_run` (吊起/底座) | sim2real空载 | FSM全流程, 关节响应正确 |
| 12 | `oceanbdx_run` (落地) | sim2real | 起立→RL站立→小速度行走 |

第11步建议先吊起机器人空腿验证RL输出是否合理, 再落地。

### 6.1 sim2sim kd 调试记录: 处理 IsaacLab 与 MuJoCo 不一致

sim2sim 的主要风险不是 ONNX 推理本身, 而是 **IsaacLab implicit actuator** 与
**MuJoCo 显式 torque PD** 的动力学差异。即使 policy 输出、观测顺序、关节顺序完全正确,
统一的 `rl_kd` 也可能导致以下现象:

- viewer 中站得住, 但零命令下缓慢横漂;
- 给 `base_link` 施加外力后, policy 有动作响应, 脚也有离地/换支撑, 但单脚支撑后拉不回来;
- 某个方向能小碎步恢复, 另一个方向变成抖动或侧翻;
- action 长时间饱和, 例如 `final_sat=10/10`, 说明控制器已经进入极限响应。

当前训练侧 actuator 配置为 `stiffness=50`, `damping=2.5`, `action_scale=0.25`。
MuJoCo sim2sim 中保留训练动作幅度 `action_scale=0.25`, 但使用 sim2sim 专用逐关节阻尼:

```yaml
sim2sim:
   action_scale: 0.25
   rl_kd: [5.0, 4.0, 8.0, 8.0, 8.0,  5.0, 4.0, 8.0, 8.0, 8.0]
```

这组参数的含义:

- `leg_*1` 髋部前后/根部关节: `kd=5`, 保留扰动恢复时的响应速度;
- `leg_*2` 髋侧摆关节: `kd=4`, 避免侧向恢复被过强阻尼锁住;
- `leg_*3/4/5` 大腿/小腿/踝相关关节: `kd=8`, 抑制 MuJoCo 显式 PD 下的高频摆腿和单脚支撑发散;
- 这组只放在 `sim2sim` 段, 不影响 C++/真机侧 `control.rl_kd`。

排查 sim2sim 不一致时, 先用 headless 固定外力测试, 不要只靠 viewer 目测:

```bash
# 0N 基线: 检查零命令自身漂移
python3 sim2sim/mujoco_sim.py \
   --debug-push-steps 112 \
   --debug-push-start 80 \
   --debug-push-duration 11 \
   --debug-push-force-x 0 \
   --debug-push-force-y 0

# 侧向 40N, 约 0.16s: 对齐 IsaacLab play 中的抗推测试量级
python3 sim2sim/mujoco_sim.py \
   --debug-push-steps 112 \
   --debug-push-start 80 \
   --debug-push-duration 11 \
   --debug-push-force-y 40

# 前向 40N
python3 sim2sim/mujoco_sim.py \
   --debug-push-steps 112 \
   --debug-push-start 80 \
   --debug-push-duration 11 \
   --debug-push-force-x 40 \
   --debug-push-force-y 0
```

`[sim_push_summary]` 中重点看:

- `final_vel_b`: 推力结束后一段时间的 body-frame 水平速度, 越接近 0 越好;
- `final_tilt_xy`: 机身最终倾斜, 若接近 `0.5` 通常已经明显失稳;
- `max_foot_z` 与 `air_steps`: 判断是否真的有抬脚/换支撑;
- `final_sat`: action 饱和数量, 长期 `8/10` 到 `10/10` 说明参数仍偏激;
- 0N 基线也要测, 因为某些 kd 对外力好, 但无外力会慢慢漂。

本次调试中的关键对照:

```text
旧 sim2sim: action_scale=0.20, kd=8 all, +Y 40N
   final_vel_y≈+0.889, final_tilt_y≈+0.519, final_sat=10/10
   左脚长时间离地, 单脚支撑后发散。

统一 kd=4 all, action_scale=0.25
   侧向 40N 可以恢复, 但 +X 前向推会诱发明显侧倾。

逐关节 kd=[5,4,8,8,8]*2, action_scale=0.25
   0N / +Y / -Y / +X / -X 短测均未发散, 速度和姿态可回收。
```

迁移新 policy 或新 URDF 时建议按这个顺序排查:

1. 先确认 ONNX 与 Isaac play 输出一致: 同一静态 obs 下 action 是否一致;
2. 确认观测顺序、关节顺序、`default_dof_pos`、`action_scale` 与训练一致;
3. 确认 IMU/projected_gravity 方向: 直立约 `[0,0,-1]`;
4. 用 0N / ±Y / ±X 固定外力 headless 测试看 `final_vel_b` 和 `final_sat`;
5. 若 policy 有明显动作但 MuJoCo 恢复失败, 优先调 sim2sim 专用 `rl_kd`;
6. 统一 kd 不够时, 使用逐关节 kd: 髋部保留响应, 膝/踝增加阻尼。

## 7. IsaacLab 训练约定

最小功能点: velocity 任务 (站立平衡 = 0速度指令, 行走 = 小速度指令)。

部署侧 (`src/policy.cpp` / `sim2sim/mujoco_sim.py`) 假定观测为:

```
[ base_ang_vel*0.25, projected_gravity, commands*(2,2,0.25),
  (dof_pos - default_dof_pos)*1.0, dof_vel*0.05, last_actions ]   共 9+3×10=39 维
动作: target_q = default_dof_pos + 0.25 * action
```

训练配置必须与 `config/oceanbdx.yaml` 的 policy 段一致 (缩放/默认关节角/kp/kd),
导出: `rsl_rl` 的 play 脚本自动导出 `exported/policy.onnx`, 放到 `policy/policy.onnx`。

注意事项:
- `init_state.joint_pos` 用站立姿态 (全0), 与 default_dof_pos 一致
- actuator 用 ImplicitActuator, stiffness/damping = rl_kp/rl_kd (50/2.5)
- 关节顺序: 核对USD解析后的 joint_names, 同步修改yaml
- 脖子4关节在训练中可固定 (fixed joint 或不暴露给policy)

## 8. 部署环境 (Jetson Orin Nano)

```bash
sudo apt install build-essential cmake libyaml-cpp-dev
# onnxruntime: 下载 aarch64 release 包 (https://github.com/microsoft/onnxruntime/releases)
tar xzf onnxruntime-linux-aarch64-*.tgz -C /opt && sudo mv /opt/onnxruntime-* /opt/onnxruntime

cd oceanbdx && mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime
make -j4

sudo cp ../config/udev/99-oceanbdx.rules /etc/udev/rules.d/   # 先按实际硬件改serial!
sudo udevadm control --reload && sudo udevadm trigger
```

实时性建议: 主程序用 `sudo chrt -f 50 ./oceanbdx_run ../config/oceanbdx.yaml`
或对串口线程设置CPU亲和性; Jetson设置 `sudo jetson_clocks` 锁定频率。

## 9. 后续扩展 (不在最小功能点内)

- 脖子FT舵机控制 (驱动已就绪: `neck_driver`, `neck_enabled: true` 激活)
- 手柄接入主控制 (驱动已就绪: `gamepad_driver` USB手柄, 参照 `tests/test_gamepad.cpp`)
- 电池监控接入主控制 (驱动已就绪: `battery_driver` A5协议, 参照 `tests/test_battery.cpp`)
- 复杂步态 / 表演动作 (BDX风格的animation重定向)
