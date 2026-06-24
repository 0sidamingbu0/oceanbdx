# OceanBDX

复刻迪士尼 BDX 的双足机器人控制工程。部署于 **Jetson Orin Nano**, 训练采用 **IsaacLab**, sim2sim 采用 **MuJoCo**。

> 本目录是一个独立工程, 可直接作为 `oceanbdx` 仓库的根目录使用。
> 驱动与URDF提取自 [sarocean](../) 工程的 `rl_real_ocean.cpp` 及其依赖。

## 整体方案

完整架构、零位标定方案、状态机设计与**分步调试路线**见 [docs/architecture.md](docs/architecture.md)。

- 双腿各一路 USB转485, 每腿 5×宇树 GO-M8010-6
- IMU: YIS320, 一路USB转串口
- 脖子: 1×M8010 + 3×飞特舵机 (驱动已移植, 控制暂不实现)
- 状态机: `PASSIVE → BOOT_CHECK(坐姿校验) → SIT_HOLD → STAND_UP(脚本) → RL_BALANCE → RL_WALK`, 任意状态可进 `DAMPING` 软停
- 电机零位=结构限位, URDF零位=站立姿态, 偏移由 `config/oceanbdx.yaml` 的 `calibration` 段管理
- 策略: IsaacLab 导出 ONNX, C++ onnxruntime 推理

## 目录结构

```
drivers/           提取的硬件驱动 (宇树M8010串口SDK / 飞特舵机 / YIS320 IMU)
include/ src/      核心控制代码 (驱动封装、USB手柄、电池、标定、FSM、策略、主程序)
tests/             分步调试程序 (单腿/IMU/舵机/手柄/电池/零位)
config/            机器人配置 + udev规则
description/       ocean URDF + meshes
sim2sim/           MuJoCo 仿真验证
scripts/           urdf2mjcf转换、零位测量工具
policy/            放置训练导出的 policy.onnx
docs/              方案架构文档
```

## 快速开始

### 编译 (Jetson / x86 Linux)

```bash
sudo apt install build-essential cmake libyaml-cpp-dev
# onnxruntime (主控制程序需要, 测试程序不需要):
#   从 https://github.com/microsoft/onnxruntime/releases 下载对应架构包解压到 /opt/onnxruntime

mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime    # 无onnxruntime时会自动跳过主程序
make -j4
```

### 分步调试 (按顺序!)

```bash
./build/test_leg_motor /dev/ttyright 5        # 1. 单腿只读
./build/test_leg_motor /dev/ttyright 5 hold   # 2. 小增益保持
./build/test_imu                              # 3. IMU
./build/test_neck                             # 4. 飞特舵机 (仅驱动验证)
./build/test_gamepad                           # 5. USB手柄 (/dev/input/js0)
./build/test_battery                           # 6. 电池BMS (/dev/ttybat, A5协议)
python3 scripts/measure_offset.py              # 7. URDF可视化测零位偏移 -> 填config
./build/test_calibration config/oceanbdx.yaml  # 8. 零位/方向校验
```

### sim2sim (MuJoCo)

```bash
pip install -r sim2sim/requirements.txt
python3 scripts/urdf2mjcf.py                  # 生成 sim2sim/ocean_scene.xml
python3 sim2sim/mujoco_sim.py --no-policy     # 仅验证起立脚本
python3 sim2sim/mujoco_sim.py                 # 加载 policy/policy.onnx 完整验证
```

sim2sim 与 IsaacLab 不一致时, 尤其是外力后单脚支撑发散、横漂或动作饱和, 先看 [docs/architecture.md](docs/architecture.md) 的 “sim2sim kd 调试记录”。

### 真机运行

```bash
sudo cp config/udev/99-oceanbdx.rules /etc/udev/rules.d/   # 按实际硬件改匹配条件
sudo udevadm control --reload && sudo udevadm trigger

cd build && sudo chrt -f 50 ./oceanbdx_run ../config/oceanbdx.yaml
# 键盘: 0=坐姿校验 1=起立 2=行走 3=回平衡 9=阻尼 p=失能 wsadqe=速度 x=停
```

## 安全须知

- **必须坐姿上电** (底座约束关节在已知范围), `BOOT_CHECK` 不通过禁止使能
- 首次RL测试先**吊起机器人**验证关节输出合理再落地
- 任何时刻按 `9` 或 Ctrl+C 进入阻尼软停

## License

Apache-2.0 (驱动各自保留原License, 见对应目录)
