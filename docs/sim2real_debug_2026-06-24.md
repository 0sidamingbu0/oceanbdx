# OceanBDX sim2real 调试记录 - 2026-06-24

本文记录 2026-06-24 的真机 RL bring-up 现象、排查结论和后续方向。用途是给之后 sim2real 继续调试时快速恢复上下文, 避免重复把正常现象当故障处理。

## 背景

当前控制链路:

```text
IsaacLab/rsl_rl policy.onnx
  -> sim2sim/mujoco_sim.py
  -> oceanbdx_run(C++ ONNX + FSM + LegDriver)
  -> Unitree M8010 真机
```

部署侧观测与动作约定:

```text
obs = [base_ang_vel*0.25, projected_gravity, commands*(2,2,0.25),
       (dof_pos-default_dof_pos), dof_vel*0.05, last_actions]
target_q = default_dof_pos + action_scale * clipped_action
```

关节顺序仍按:

```text
[leg_r1, leg_r2, leg_r3, leg_r4, leg_r5, leg_l1, leg_l2, leg_l3, leg_l4, leg_l5]
```

## 当天现象

### 吊起测试

真机按键流程正常:

```text
BOOT_CHECK -> SIT_HOLD -> STAND_UP -> RL_BALANCE
```

传感器和总线频率稳定:

```text
imu_hz ~= 199-200
left_hz ~= 119-120
right_hz ~= 119-120
projected_gravity ~= (0, 0, -1)
gyro ~= 0
```

进入 `RL_BALANCE` 后, 机器人腿部只摆动几下后基本不动。此时没有掉进 `DAMPING`, 状态机仍在 `RL_BALANCE`。

### 落地测试

放到地上后, RL 接管后双腿持续交替跳动。主观表现像高频扰动/噪声驱动下的步态抖动, 但不是明显力矩不足:

- `kp/kd` 量级能提供站住的刚度;
- 电机输出力矩感觉接近需求;
- 起立脚本和状态机切换没有明显异常;
- 整体 sim2real 链路初步看没有大方向错误。

当前判断: 真机链路已进入“策略/仿真质量决定表现”的阶段, 后续优先回到训练和 sim2sim 调教, 暂停继续硬调 sim2real。

## 关键排查结论

### 1. action 饱和不是本策略的故障判据

一开始在吊起自检中看到:

```text
policy_act_absmax=1
```

曾误判为部署观测或 policy 导出异常。后来确认:

- IsaacSim 中也会出现关节/action 饱和;
- 当前 sim2sim 中也能在 action 饱和条件下正常工作;
- 因此 `policy_act_absmax=1` 不能单独作为错误判据。

以后看自检时, `policy_act_absmax=1` 只代表 policy 输出顶到训练侧 clip, 需要结合 `q/dq`、`rl_target`、姿态、接触和实际运动判断。

### 2. 上一版“保守真机参数”会把 RL 输出压得过小

曾将真机参数改为更保守:

```yaml
rl_kp: 20
rl_kd: 1.5
torque_limits: 8
policy.action_scale: 0.10
rl_warmup_duration: 2.0
rl_target_rate_limit: 1.0
```

这会造成真机和 sim2sim 明显不一致:

- sim2sim 使用 `action_scale=0.25`;
- sim2sim 没有 C++ FSM 中新增的 warmup/rate limit;
- `action_scale=0.10` 会把饱和 action 的最大目标角从 `0.25 rad` 压到 `0.10 rad`;
- 2 秒 warmup 叠加 `1 rad/s` 目标角限速后, RL 接管初期动作更小, 容易表现为“摆两下就不动”。

因此当天已把真机主控参数恢复到与训练/sim2sim 更一致:

```yaml
rl_kp: 50
rl_kd: 2.5
torque_limits: 23
policy.action_scale: 0.25
rl_warmup_duration: 0.0
rl_target_rate_limit: 0.0
```

注: 这不是最终真机安全参数, 只是为了排除“部署侧额外限制导致行为不同”。之后如果要重新加安全限制, 需要先在 sim2sim 中同步复现对应限制。

### 3. sim2real 基础链路目前没有明显大错

从今天日志看, 以下链路基本可认为通过初步检查:

- FSM: `SIT_HOLD -> STAND_UP -> RL_BALANCE` 正常;
- IMU: 直立/吊平时 projected gravity 方向正确, 接近 `(0,0,-1)`;
- IMU 频率: 约 200Hz, 稳定;
- 电机反馈频率: 两腿约 120Hz, 稳定;
- 起立脚本: 能从坐姿起到站姿;
- 标定量级: `q-default` 在站立附近, 没有明显全局符号反向或数量级错误;
- policy 加载和推理: ONNX 输入输出维度正确, 持续有动作输出。

仍需在下一轮继续留意:

- 真机关节速度噪声是否会放大 policy 抖动;
- 地面接触和足端摩擦是否与 sim2sim 差异过大;
- 左右腿机械间隙、地面不平、足底材料是否诱发交替跳动;
- 电机侧实际 PD/力矩限幅与 MuJoCo 显式 PD 的差异。

## 已加入的诊断输出

为方便后续定位, C++ 主控保留了更详细的自检输出。

启动时会打印 zero-stand probe:

```text
[Policy] zero-stand probe: act_absmax=... raw_act_absmax=... saturated=N/10
```

该输出只用于记录 policy 在理想静态输入下的数值, 不再把饱和作为 warning。

运行中按 `c` 自检会打印:

```text
projected_gravity
gyro
q-default per joint
q_dev_absmax
dq_absmax
policy_act_absmax
raw_act_absmax
action
raw_action
rl_target
target-q
policy_path/action_scale/rl_target_rate_limit/rl_warmup_duration
```

解释重点:

- `raw_action`: ONNX 原始输出, 未经过 `clip_actions`;
- `action`: clip 后动作, 即用于生成 target 的动作;
- `rl_target`: 当前 FSM 最终发出的 RL 目标角, 已经过 warmup/rate limit/软限位;
- `target-q`: 目标角与当前关节角误差, 可判断电机是否跟得上。

## 后续优先级

### 优先回到训练和 sim2sim

当前落地现象是 RL 后双腿交替跳动, 而 sim2sim 也还有抖动。下一步应优先把 sim2sim 和训练调好, 再继续真机上地。

建议顺序:

1. 在 IsaacSim/IsaacLab 中确认站立和零速度命令下的接触、足端高度、基座高度、关节速度噪声是否合理。
2. 在 sim2sim 中复现落地双腿交替跳动, 不要只看是否摔倒, 要看 `dq`、接触力、足端离地、左右支撑切换频率。
3. 先把 0 速度站立调稳, 再测小速度行走。
4. 将任何真机安全限制, 如 action_scale 降低、target rate limit、warmup, 先同步加到 sim2sim 中验证。
5. sim2sim 稳定后再上真机吊起, 最后短时间落地测试。

### sim2sim 需要关注的指标

建议保留/增强以下 debug:

```text
base height / projected gravity / body velocity
per-joint q, dq, target-q
raw_action/action 饱和分布
left/right foot contact force
foot air time / contact switching frequency
PD torque 或等效关节力矩
```

对当前“交替跳动”现象, 重点看:

- 零命令时是否也有左右脚周期性交替离地;
- `dq_absmax` 是否先升高再驱动 action 饱和;
- contact force 是否在左右脚之间高频切换;
- target-q 是否持续保持固定偏差, 还是目标本身在高频反转;
- 抖动是否由踝/膝关节主导, 还是髋侧摆主导。

### 训练侧可尝试方向

以下是候选方向, 需要在训练和 sim2sim 中验证, 不应直接在真机上盲调:

- 增强 0 速度站立稳定性奖励, 减少无命令下主动换步;
- 调整 action rate / joint velocity 惩罚, 抑制高频腿部交替动作;
- 检查 command sampling 中 0 速度比例是否足够;
- 检查 feet air time / gait 相关奖励是否在 0 速度下仍鼓励抬脚;
- 引入或加强 joint position limit、torque、action smoothness 惩罚;
- 训练中加入更接近 MuJoCo/真机的 actuator delay、PD、摩擦、质量和 IMU/关节噪声随机化;
- 确认 default pose 和站立高度让双足接触几何处在自然静态平衡点, 避免 policy 必须靠小跳维持。

## 下一轮真机前检查清单

在重新落地前, 建议满足:

- sim2sim 零速度站立不再有明显持续交替跳动;
- sim2sim 中 `dq_absmax` 和足端接触切换频率处于可接受范围;
- C++ 与 sim2sim 的 `action_scale`、warmup、target rate limit、kp/kd、torque limit 是否有差异已明确记录;
- 吊起自检中 `projected_gravity`、`gyro`、`q-default`、`target-q` 符合预期;
- 落地首次测试只保持短时间, 手随时准备 `9` 或空格进入 `DAMPING`。

## 当天结论

今天的主要收获是: sim2real 链路已经不像是存在观测维度、IMU 方向、关节顺序、状态机切换这类大错。落地后的交替跳动更像策略和仿真本身尚未足够稳, 需要回到训练和 sim2sim 把零速度站立与接触行为调顺。

下一阶段目标: 先让 sim2sim 在 0 速度下稳定、低抖动、不主动交替跳步, 再继续 sim2real。
