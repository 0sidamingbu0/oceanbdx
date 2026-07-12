# OceanBDX

复刻迪士尼 BDX 的双足机器人控制工程。部署于 **Jetson Orin Nano**, 训练采用 **IsaacLab**, sim2sim 采用 **MuJoCo**。

> 本目录是一个独立工程, 可直接作为 `oceanbdx` 仓库的根目录使用。
> 驱动与URDF提取自 [sarocean](../) 工程的 `rl_real_ocean.cpp` 及其依赖。

## 整体方案

完整架构、零位标定方案、状态机设计与**分步调试路线**见 [docs/architecture.md](docs/architecture.md)。

- 双腿各一路 USB转485, 每腿 5×宇树 GO-M8010-6
- IMU: YIS320, 一路USB转串口
- 脖子: 1×M8010 + 3×飞特舵机 (驱动已移植, 控制暂不实现)
- 状态机: `PASSIVE → SIT_ALIGN(脚本回蹲姿) → BOOT_CHECK(坐姿校验) → SIT_HOLD → STAND_UP(脚本) → RL_STAND(站立模型) ⇄ RL_WALK(行走模型)`, 任意状态可进 `DAMPING` 软停
- 电机零位=结构限位, URDF零位=站立姿态, 偏移由 `config/oceanbdx.yaml` 的 `calibration` 段管理
- 策略: 论文 divide-and-conquer **两个独立 ONNX**——站立 `policy/stand/policy.onnx`（77维观测）与行走 `policy/policy.onnx`（80维观测），均输出14维动作；IsaacLab 导出，Python MuJoCo sim2sim 已支持双模型切换，C++ 真机主控仍待升级

两策略切换时保留最近两帧归一化动作、当前腿/脖子目标、path frame 和 FOH/低通状态；
`RL_WALK → RL_STAND` 会等待下一次确认的双支撑，避免在摆动相中途换策略。站立和行走统一
使用论文附录 B 的腿部软件 PD `kP=10`、`kD=0.3`，不能再给站立模型单独使用旧高增益 plant。
walk→stand 会用切换瞬间的实测躯干姿态初始化 standing torso 命令；`STAND_UP→RL_STAND`
也先从实测躯干姿态起步，再用 `0.5s` 余弦接管平滑回用户/neutral 命令，同时渐入腿/脖子
策略目标并把脚本 `50/3` 平滑降到 RL `10/0.3`。

## 目录结构

```
drivers/           提取的硬件驱动 (宇树M8010串口SDK / 飞特舵机 / YIS320 IMU)
include/ src/      核心控制代码 (驱动封装、USB手柄、电池、标定、FSM、策略、主程序)
tests/             分步调试程序 (单腿/IMU/舵机/手柄/电池/零位)
config/            机器人配置 + udev规则
description/       ocean URDF + meshes
sim2sim/           MuJoCo 仿真验证
scripts/           urdf2mjcf转换、零位测量工具
policy/            训练导出的 ONNX: policy.onnx(行走) + stand/policy.onnx(站立)
docs/              方案架构文档
changelog/         每日修改履历 (见下方约定)
```

## 修改履历 (changelog)

每天的改动记录统一放在 [changelog/](changelog/) 文件夹, 一天一个文件, 命名 `YYYY-MM-DD.md`。
目的: 留存历史上下文, 便于自己回溯, 也便于其他工程/协作者快速理解本工程的演进履历。

每篇建议写清: 背景/现象、涉及文件、改了什么、为什么这么改、验证结论、遗留待办。
跨工程的根因排查 (如 sim2sim 与 IsaacLab 训练侧联调) 也记在当天文件里。

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
python3 sim2sim/mujoco_sim.py                 # 同时加载行走+站立两个 ONNX 完整验证
# 站立/行走是两个独立模型: 起立(1)后进站立模型, 按 2 切行走模型, 按 1 切回站立模型
python3 sim2sim/mujoco_sim.py --stand-policy policy/stand/policy.onnx   # 覆盖站立 onnx 路径
python3 sim2sim/diag_walk_policy.py --vx 0.15    # 无界面前进逐脚间隙/力矩/脖子诊断
python3 sim2sim/diag_walk_policy.py --vx -0.15   # 同口径后退诊断
```

当前 Python sim2sim 会严格检查 stand/walk 输入维度为 `77/80`，旧 74 维站立和 77 维行走
ONNX 会直接拒绝加载。数字键请求在主控制循环边界消费；行走回站立时，移动状态下会先经历
单支撑并等待下一双支撑，零速已双脚着地时可直接完成切换，超时则取消请求。path-frame FK
使用脚 link body origin/quaternion 对齐 IsaacLab，sole geom center 只用于接触；旧算法在 neutral
pose 会产生约 `2.585cm` 的位置偏差。

> **跑 IsaacLab / 需要 torch 的脚本时(如 `sim2sim/scan_ckpt.py`、`export_ckpt_onnx.py`),用 IsaacLab 运行时 python:**
> ```bash
> /home/ocean/oceanisaaclab/oceanisaaclab/_isaaclab/isaaclab.sh -p <脚本.py> [参数]
> ```
> 该 python 同时具备 torch + mujoco + onnxruntime,避免再去折腾 conda 环境(base 缺 torch、sar 缺 mujoco)。
> `mujoco_sim.py --probe-policy` / `--debug-push-steps` 等纯推理工具用 base 的 `python3` 即可(只需 mujoco+onnxruntime)。

键盘 (★聚焦运行脚本的**终端窗口**操作, 与真机 main.cpp 一致, 不会触发 MuJoCo 自带快捷键):
`0`蹲姿 `1`起立/切站立模型 `2`切行走模型 `3`切站立模型(同1) `9`阻尼 `r`重置 `p`真机电机开关；
`w/s`=vx± `a/d`=vy± `q/e`=wz± `x`速度清零 (速度仅在 `2` 行走模型态生效)；
`t/g`=躯干pitch± `v/c`=躯干yaw± `y/b`=躯干roll± `f/z`=躯干高度± (仅 `1` 站立模型态生效)；
`i/k`=点头 `j/l`=摇头 `u/o`=歪头 `n/m`=头高 `h`=头命令清零 (站立/行走均生效)。
MuJoCo 窗口内也能按 (方向键映射到 w/s/q/e), 但数字键会附带触发 MuJoCo 的 geomgroup 切换, 故脚本每帧复位可见组防止机器人消失; 推荐还是用终端操作。

站立 torso 命令会限幅到训练可行域：h `[-0.04,+0.01]m`、pitch `±0.17rad`、yaw
`±0.24rad`、roll `±0.09rad`。不要扩大部署限幅让策略外推。

sim2sim 与 IsaacLab 不一致时, 尤其是外力后单脚支撑发散、横漂或动作饱和, 先看 [docs/architecture.md](docs/architecture.md) 的 “sim2sim kd 调试记录”。

### 真机运行

> 当前 `src/` 下 C++ 主控仍是旧 41 维、10 动作、单策略观测链路，尚未实现本文的
> stand77/walk80 双模型闭环。`sim2sim/mujoco_sim.py --real` 只是把 MuJoCo 闭环生成的目标角
> 通过 UDP/485 桥发送到真机，策略观测仍来自 MuJoCo，不是真机 IMU、关节状态和估计器驱动的
> 完整闭环，也不能作为真机无缝切换已经完成的依据。

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
