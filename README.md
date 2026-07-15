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

两策略切换时保留最近两帧归一化动作、当前腿/脖子目标、path frame 和 FOH/低通状态；这些
低层连续状态只用于避免首帧目标角/力矩跳变，不代表新策略继续保持旧策略的姿态命令。
`RL_WALK → RL_STAND` 由 walking policy 依次完成 `0.6s` 余弦减速、参考双支撑相位归零、
`0.15s` 最终平滑过零、`0.4s` 零命令收脚和连续 `0.2s` 稳定确认，再切到 standing policy。
正式放行只使用内部
步态相位、IMU projected gravity/gyro、腿 `q/dq` 和当前 policy target，不依赖脚底力传感器、
MuJoCo 真值速度或 CoM。超时后 walking policy 保持零命令，不冒险强制切换。

`RL_STAND → RL_WALK` 不再从蹲姿或倾斜姿态直接切模型。standing policy 先将当前 torso 和 head 命令
用同一条半余弦曲线平滑回到 neutral，回正时长按两者最大归一化幅度缩放，最大 `1.5s`；neutral 后至少保持 `0.3s`，
再用同一组 IMU、腿 `q/dq` 和当前 target 连续确认稳定 `0.2s`，最后才切入 walking policy
零速闭环。回正期间屏蔽新的 head/torso 命令；进入 walking 后 head/torso 命令从零开始，不恢复旧姿态。
切换请求取消后仍继续平滑回 neutral，不恢复旧蹲姿；总计 `5s` 超时则留在
standing policy neutral，不强制切换，也不关闭电机输出。

站立和行走统一使用论文附录 B 的腿部软件 PD `kP=10`、`kD=0.3`，不能再给站立模型单独
使用旧高增益 plant。模型互切不恢复旧姿态：进入 standing 时 torso 命令归零并保持 neutral；
请求进入 walking 时先由 standing policy 平滑回 neutral。键盘手动切换仍将速度目标归零；
手柄模式切换保留当前摇杆目标，但 effective command 从零限加速度建立。两个模型语义相同的
head 命令继续保留。`STAND_UP→RL_STAND`
仍使用原有 `0.5s` 接管窗渐入腿/脖子策略目标，并把脚本 `50/3` 平滑降到 RL `10/0.3`。

头部命令语义相同，但限幅按当前实际运行的 policy 独立选择：standing 使用完整表现域
Δh `±0.02m`、pitch `±0.50rad`、yaw `±1.00rad`、roll `±0.60rad`；walking 为抑制行进甩头
继续使用约 1/3 域 `±0.007m / ±0.17 / ±0.33 / ±0.20rad`。手柄映射按当前 policy 限幅；站转走
回正完成后 head 命令从零重新接收输入，因此 walking 永不接收超出训练域的输入，也不会恢复
站立时的旧头部姿态。切换未完成时不会提前采用目标模型的限幅。

遥控逻辑对齐论文附录 C / 表 VII：遥控模式启动时默认请求 standing，起立完成后进入 standing
policy；standing 中按下 `R1` 立即请求行走，walking 中短按切回站立；行走中长按 `R1`
将速度增益从默认 `50%` 提高到 `100%`，`A` 明确请求站立。
摇杆回中只表示零速度，不会自动退出 walking policy。站立时左摇杆控制躯干姿态、右摇杆控制
头部并在视线极限处加入躯干转动；行走时左摇杆改为前进/转向，`L2/R2` 改为横移，右摇杆
继续控制头部。也可把 YAML 的 `walk_button_behavior` 改为 `hold`，实现按住 `R1` 才行走；
该适配默认仍使用 `50%` 速度增益。由于本手柄的 `L2/R2` 是数字按键，普通档理论横移
`0.15*50%=0.075m/s` 会落入策略的 `0.08` 零速区；部署映射将其提升为训练域内的
`0.09m/s`，长按 `R1` 时仍使用最大 `0.15m/s`。论文原机的横移上限为 `0.4m/s`，其 50%
仍有 `0.2m/s`，不存在该冲突；这里是本项目缩小训练速度域后的必要适配。

站走过渡期间，摇杆始终按当前正在运行的 policy 解释，直到模型真正切换；因此停走减速时
前进杆不会提前变成 torso pitch。进入 standing 后，左摇杆和 `L2/R2` 必须连续回中 `0.15s`
才重新使能身体姿态控制，避免未回中的行走命令在切换瞬间变成大幅前倾/侧倾命令。`A` 的
standing 请求优先于任何重叠的 `R1` 手势。

本项目实测手柄的 `START` 是 `button[11]`，已映射为与键盘 `1` 相同：仿真坐姿时启动
`STAND_UP` 起立流程；处于 walking 时请求安全减速并切回 standing；已经 standing 时保持不变。

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
python3 sim2sim/mujoco_sim.py --gamepad       # 启用论文 R1/双摇杆遥控映射 (/dev/input/js0)
# 站立/行走是两个独立模型: 起立(1)后进站立模型, 按 2 切行走模型, 按 1 切回站立模型
python3 sim2sim/mujoco_sim.py --stand-policy policy/stand/policy.onnx   # 覆盖站立 onnx 路径
python3 sim2sim/mujoco_sim.py --viewer-push-force-y 60 --viewer-push-duration 0.1  # viewer定量侧推
python3 sim2sim/diag_walk_policy.py --vx 0.15    # 无界面前进逐脚间隙/力矩/脖子诊断
python3 sim2sim/diag_walk_policy.py --vx -0.15   # 同口径后退诊断
python3 -m unittest -v sim2sim/test_walk_stand_switch.py  # 切换/命令平滑回归
python3 -m unittest -v sim2sim/test_gamepad_mapping.py    # 论文遥控器映射回归
python3 -m unittest -v sim2sim/test_terrain_scene.py      # 地形几何/接触/动力学参数回归
```

`--gamepad` 会持续等待并支持热插拔 `/dev/input/js0`。看到终端输出 `[gamepad] connected` 后，
按手柄 `START` 即可代替键盘 `1` 起立，之后可完全使用上述手柄映射操作。

可视化定量推力使用世界坐标系 `--viewer-push-force-x/y/z`，作用点为 `base_link` 质心；进入
`RL_STAND` 或 `RL_WALK` 后，在运行脚本的终端短按 `5` 施加配置方向，短按 `6` 施加反方向，可重复
触发。默认持续 `0.1s`，与训练 Table V 大推持续时间一致；侧推建议从 `±30N` 开始，再测试
`±60N / ±80N / ±95N`。该功能在 `--real` 模式下强制禁用；MuJoCo 窗口原有的双击选中后
`Ctrl+右键` 鼠标拖动仍可用于不定量测试。

场景保留原点附近的平地出生区，并提供两个颜色区分的测试场：机器人初始朝世界 `-X`，绿色
坡道位于正前方 `x=[-4.2,-1.2]m`，直接按 `w` 可依次测试 `5°` 上坡、`1m` 平台和 `5°`
下坡；棕灰色粗糙区位于初始左侧、中心 `(0,-2.1)m`，按 `a` 可横移进入，也可先左转约
`90°` 再正向走入。粗糙区为 `2.5m × 2.5m`、`5cm` 网格和 `4mm` 量化高度，保持训练配置
的 `24mm` 峰谷差；为避免无限平面截掉负高度，整体平移为 `0..24mm`，外围 `0.25m` 平滑
过渡到平地。viewer 默认跟随 `base_link`，进入远端区域后仍保持机器人在画面中。

当前 Python sim2sim 会严格检查 stand/walk 输入维度为 `77/80`，旧 74 维站立和 77 维行走
ONNX 会直接拒绝加载。数字键请求在主控制循环边界消费；行走回站立时，速度命令与用户命令
分离，减速阶段仍保持相位推进，只在训练参考的双支撑相位窗口将策略命令归零。稳定判据任意
一帧失败都会重置确认计时，`3.5s` 超时后继续由 walking policy 以零命令保持；再次请求行走时
由统一限加速度器平滑跟踪当前用户命令。普通加速、减速、清零和反向也使用同一控制线程平滑器；
清零或全向反转只有在参考双支撑窗口才允许穿过移动阈值，避免相位冻结在单支撑。
手柄模式请求同样只在 200Hz 控制线程消费：`R1` 请求停止后复用完整 walk→stand 安全流程；
过渡中再次请求行走会从当前 effective command 平滑恢复。稳定 neutral 站立已提前累计稳定时间，
所以 `R1` 请求行走可在下一策略边界切模型；当前摇杆速度目标会保留，但送入 walking policy 的
effective command 仍从零按限加速度建立。切换超时会锁存失败目标，避免相同输入自动循环重试。
显式按 `A` 或手柄断连边沿可解除一次 standing 超时锁存并重新执行安全停站；持续不变的输入
不会循环重试或刷日志。
path-frame FK 使用脚 link body origin/quaternion 对齐 IsaacLab，
sole geom center 只用于接触诊断；旧算法在 neutral pose 会产生约 `2.585cm` 的位置偏差。
脚底接触力诊断会同时识别平地、坡道和粗糙区；离地间隙按每个脚底 box 角点下方的局部地形
高度计算，不再把所有区域错误地当作世界 `z=0`。

> **跑 IsaacLab / 需要 torch 的脚本时(如 `sim2sim/scan_ckpt.py`、`export_ckpt_onnx.py`),用 IsaacLab 运行时 python:**
> ```bash
> /home/ocean/oceanisaaclab/oceanisaaclab/_isaaclab/isaaclab.sh -p <脚本.py> [参数]
> ```
> 该 python 同时具备 torch + mujoco + onnxruntime,避免再去折腾 conda 环境(base 缺 torch、sar 缺 mujoco)。
> `mujoco_sim.py --probe-policy` / `--debug-push-steps` 等纯推理工具用 base 的 `python3` 即可(只需 mujoco+onnxruntime)。

键盘 (★聚焦运行脚本的**终端窗口**操作, 与真机 main.cpp 一致, 不会触发 MuJoCo 自带快捷键):
`0`蹲姿 `1`起立/切站立模型 `2`切行走模型 `3`切站立模型(同1) `5/6` viewer定量推力正/反向；
`9`阻尼 `r`重置 `p`真机电机开关；
`w/s`=vx± `a/d`=vy± `q/e`=wz± `x`速度清零 (速度仅在 `2` 行走模型态生效)；
`t/g`=躯干pitch± `v/c`=躯干yaw± `y/b`=躯干roll± `f/z`=躯干高度± (仅 `1` 站立模型态生效)；
`i/k`=点头 `j/l`=摇头 `u/o`=歪头 `n/m`=头高 `h`=头命令清零 (站立/行走均生效)。
MuJoCo 窗口内也能按 (方向键映射到 w/s/q/e), 但数字键会附带触发 MuJoCo 的 geomgroup 切换, 故脚本每帧复位可见组防止机器人消失; 推荐还是用终端操作。

键盘以及手柄行走模式写入的是速度目标，不会直接跳变 policy 命令。默认限加速度为 vx/vy/wz
`[0.50,0.40,1.50]`，限减速度为 `[0.40,0.30,1.20]`，单位分别为
`[m/s²,m/s²,rad/s²]`，可在 YAML 的 `command` 段调整。

论文手柄默认映射见 YAML `puppeteering` 段。不同 USB 手柄的 Linux axis/button 编号并不统一，
更换设备后先运行 `./build/test_gamepad /dev/input/js0` 核对；断开设备时 Python 输入会立即归零并
请求站立，C++ `GamepadDriver::GetState()` 也不会再返回拔出前遗留的摇杆值。

站立 torso 命令会限幅到训练可行域：h `[-0.04,+0.01]m`、pitch `±0.17rad`、yaw
`±0.24rad`、roll `±0.09rad`。head 的 stand/walk 限幅分别由 `stand_max_head_*` 和
`max_head_*` 配置。此前的 77 维 stand ONNX 只见过缩小后的 head 域；即使输入形状仍匹配，
也不能直接使用完整站立限幅。必须在 OceanIsaacLab 用新的联合 head+torso 参考、命令分布和
clean/disturbed 训练分层从头训练、重新导出后，才替换 `policy/stand/policy.onnx[.data]`。
当前训练已删除会改变奖励权重的 recovery gate，只保留基于 IMU、状态估计和接触的诊断指标；
不要单独扩大部署限幅让旧策略外推。

sim2sim 与 IsaacLab 不一致时, 尤其是外力后单脚支撑发散、横漂或动作饱和, 先看 [docs/architecture.md](docs/architecture.md) 的 “sim2sim kd 调试记录”。

### 真机运行

> 当前 `src/` 下 C++ 主控仍是旧 41 维、10 动作、单策略观测链路，尚未实现本文的
> stand77/walk80 双模型闭环。`sim2sim/mujoco_sim.py --real` 只是把 MuJoCo 闭环生成的目标角
> 通过 UDP/485 桥发送到真机，策略观测仍来自 MuJoCo，不是真机 IMU、关节状态和估计器驱动的
> 完整闭环，也不能作为真机无缝切换已经完成的依据。
> 本次 `--gamepad` 映射和双模型切换仍位于 Python sim2sim；C++ 侧本轮只修复了手柄断连快照清零。

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
