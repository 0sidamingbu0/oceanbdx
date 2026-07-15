#!/usr/bin/env python3
"""
OceanBDX sim2sim: MuJoCo + ONNX policy 验证

复刻真实机器人的状态机最小闭环: SIT -> STAND_UP(脚本插值) -> RL_STAND(站立模型) <-> RL_WALK(行走模型)
★ 站立与行走是两个独立训练/导出的 ONNX 模型 (policy/stand/policy.onnx 与 policy/policy.onnx)：
  - 起立完成后自动进入 RL_STAND, 加载站立模型 (77 维观测, torso 位姿命令, 无相位);
  - 按 2 切到行走模型 (RL_WALK, 80 维观测, 速度命令 + 相位); 按 1 切回站立模型。
观测/动作处理对齐 OceanIsaacLab 训练侧；C++ 真机主控仍待升级到双模型接口。这里用于验证:
  - IsaacLab 导出的两个 policy.onnx 正确性
  - 观测顺序/缩放/默认关节角配置正确性
  - 起立脚本与RL切换的衔接、站立/行走模型互切

用法 (在 oceanbdx 根目录):
    python3 sim2sim/mujoco_sim.py [--policy policy/policy.onnx] [--config config/oceanbdx.yaml]
    python3 sim2sim/mujoco_sim.py --gamepad       # 使用论文附录 C 的 R1/双摇杆映射
    python3 sim2sim/mujoco_sim.py --viewer-push-force-y 60 --viewer-push-duration 0.1
    python3 sim2sim/mujoco_sim.py --no-policy     # 无策略, 只验证起立脚本与RL接管状态机
    python3 sim2sim/mujoco_sim.py --manual        # 纯sim: 拖动滑块摆关节角
    python3 sim2sim/mujoco_sim.py --real --no-policy # 仿真目标角→真机, 面板显示真机误差
    python3 sim2sim/mujoco_sim.py --real --manual # 真机联调: 拖动滑块→PD下发到真机
    # ★ --real 需先安装 unitree_actuator_sdk, 且停掉 C++ 主控 oceanbdx_run (串口互斥)
    # 默认每次 policy 推理把 观测+raw/clip动作+目标角 写入 runlog/sim2sim_<时间戳>.csv；
    # 用 --no-log 关闭 CSV 记录。

键盘 (★推荐聚焦“终端窗口”操作, 与真机 main.cpp 一致, 不触发 MuJoCo 快捷键):
    0 = 真机缓慢到蹲姿   1 = 起立/切站立模型   2 = 切行走模型   3 = 切站立模型(同1)   9 = 阻尼   r = 重置
    5/6 = viewer 配置推力正向/反向 (仅仿真, 在终端短按)
    p = 真机电机输出开关
    w/s = vx±0.1   a/d = vy±0.1   q/e = wz±0.1   x = 速度清零
    (未启用 --gamepad 时，速度指令仅在 RL_WALK 行走模型生效，先按 2 进入行走)
    t/g = 躯干pitch±   v/c = 躯干yaw±   y/b = 躯干roll±   f/z = 躯干高度±
    (躯干位姿命令仅在 RL_STAND 站立模型生效, 先按 1 进入站立)
    MuJoCo 窗口内也可按同样的键(方向键映射到 w/s/q/e), 但数字键会附带触发
    MuJoCo 自带 geomgroup 切换, 已每帧复位防止机器人 geom 被隐藏。
"""
import argparse
import os
import queue
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

try:
    from .gamepad_input import GamepadSnapshot, LinuxJoystick, PuppeteeringMapper
except ImportError:
    from gamepad_input import GamepadSnapshot, LinuxJoystick, PuppeteeringMapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(ROOT, "sim2sim/ocean_scene.xml")

LEG_JOINTS = ["leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
              "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint"]
NECK_JOINTS = ["neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint"]
# 动作向量顺序 = 腿 10 + 脖子 4（与训练侧 action 布局一致），调试打印按此索引
ACTION_JOINTS = LEG_JOINTS + NECK_JOINTS
GROUND_GEOM_NAMES = ("floor", "slope_terrain", "rough_terrain")


def quat_rotate_inverse_gravity(q):
    """projected gravity = quat_rotate_inverse(q, (0,0,-1)), q=(w,x,y,z). 与 policy.cpp 一致"""
    w, x, y, z = q
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ])


def yaw_from_quat(q):
    """从四元数 (w,x,y,z) 取 body +x 在世界系的 yaw（绕 z），与 euler_xyz 的 yaw 分量一致。"""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_from_quat(q):
    """Return intrinsic XYZ roll, pitch and yaw for a MuJoCo (w,x,y,z) quaternion."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.array([roll, pitch, yaw_from_quat(q)], dtype=float)


def wrap_angle(a):
    """把角度环绕到 (-π, π]。与训练侧 path_frame.wrap_angle 一致。"""
    return np.arctan2(np.sin(a), np.cos(a))


class PathFrameNP:
    """path frame（BDX 论文 V-A / Fig.4）的单 env numpy 复刻。

    逐行为对齐训练侧 oceanisaaclab .../path_frame.py：+x 轴=头部前向，行走按 path 系
    命令积分、站立一阶低通收敛到双脚中心+双脚平均朝向、最大偏差投影拉回躯干附近。
    sim2sim 用 MuJoCo 真值驱动（躯干世界 xy/yaw 里程计 + 双脚 FK 中心/朝向）；真机侧改由
    状态估计器提供同样的量（论文 V-D 要求 runtime 与训练逐行为一致）。
    """

    def __init__(self, stand_time_constant=1.0, max_pos_deviation=0.25, max_yaw_deviation=0.6):
        self.pos = np.zeros(2)   # 世界系 xy
        self.yaw = 0.0           # 头部前向朝向（世界系）
        self.stand_time_constant = float(stand_time_constant)
        self.max_pos_deviation = float(max_pos_deviation)
        self.max_yaw_deviation = float(max_yaw_deviation)

    def reset(self, base_pos_xy, head_yaw):
        self.pos = np.asarray(base_pos_xy, dtype=float).copy()
        self.yaw = float(head_yaw)

    def step(self, dt, cmd, moving, base_pos_xy, head_yaw, feet_center_xy, feet_heading_yaw):
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        if moving:
            # 行走：path 系命令 (vx 头前, vy 头左, wz) 旋到世界系再积分
            dx_w = cmd[0] * cos_y - cmd[1] * sin_y
            dy_w = cmd[0] * sin_y + cmd[1] * cos_y
            self.pos = self.pos + np.array([dx_w, dy_w]) * dt
            self.yaw = self.yaw + cmd[2] * dt
        else:
            # 站立：一阶低通收敛到双脚平均位置与朝向（论文 Fig. 4）。
            alpha = min(1.0, dt / self.stand_time_constant)
            self.pos = self.pos + alpha * (np.asarray(feet_center_xy, dtype=float) - self.pos)
            self.yaw = self.yaw + alpha * wrap_angle(float(feet_heading_yaw) - self.yaw)
        self.yaw = wrap_angle(self.yaw)
        # 最大偏差投影：把 path frame 拉回躯干附近（位置 + 朝向分别投影）
        offset = self.pos - np.asarray(base_pos_xy, dtype=float)
        dist = float(np.linalg.norm(offset))
        scale = min(1.0, self.max_pos_deviation / max(dist, 1e-6))
        self.pos = np.asarray(base_pos_xy, dtype=float) + offset * scale
        yaw_err = wrap_angle(self.yaw - head_yaw)
        self.yaw = wrap_angle(head_yaw + np.clip(yaw_err, -self.max_yaw_deviation, self.max_yaw_deviation))

    def base_in_path_frame(self, base_pos_xy, head_yaw):
        """躯干在 path 系中的 xy 位置 (2,) 与相对 yaw（观测用）。"""
        rel = np.asarray(base_pos_xy, dtype=float) - self.pos
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        x_pf = rel[0] * cos_y + rel[1] * sin_y
        y_pf = -rel[0] * sin_y + rel[1] * cos_y
        return np.array([x_pf, y_pf]), wrap_angle(head_yaw - self.yaw)


class TeleopPanel:
    """Tkinter 关节角拖动面板: 每个关节一个滑块, 实时显示真机读数。

    与 MuJoCo viewer 并存 (独立窗口), 在主循环里用 update() 非阻塞刷新。
    """

    def __init__(self, names, lower, upper, init_q, on_sync=None):
        import tkinter as tk
        self.tk = tk
        self.on_sync = on_sync
        self.root = tk.Tk()
        self.root.title("OceanBDX 关节联调")

        self.enabled = tk.BooleanVar(value=False)
        hdr = tk.Frame(self.root)
        hdr.pack(fill="x", padx=6, pady=4)
        tk.Checkbutton(hdr, text="使能电机 (PD跟随滑块)", variable=self.enabled,
                       fg="red").pack(side="left")
        tk.Button(hdr, text="滑块←当前角", command=self._sync).pack(side="left", padx=8)

        self.scales = []
        self.real_lbls = []
        for i, name in enumerate(names):
            row = tk.Frame(self.root)
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=name, width=12, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=float(init_q[i]))
            tk.Scale(row, from_=float(lower[i]), to=float(upper[i]), resolution=0.001,
                     orient="horizontal", length=240, variable=var).pack(side="left")
            lbl = tk.Label(row, text="real: --", width=16, anchor="w")
            lbl.pack(side="left")
            self.scales.append(var)
            self.real_lbls.append(lbl)

        self._alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        self._alive = False

    def _sync(self):
        if self.on_sync is None:
            return
        q = self.on_sync()
        if q is not None:
            for i, var in enumerate(self.scales):
                var.set(float(q[i]))

    def get_targets(self):
        return np.array([v.get() for v in self.scales])

    def set_real(self, q, tau=None):
        for i, lbl in enumerate(self.real_lbls):
            t = "" if tau is None else f" {tau[i]:+.1f}Nm"
            lbl.config(text=f"real:{q[i]:+.3f}{t}")

    def is_enabled(self):
        return bool(self.enabled.get())

    def update(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self._alive = False

    def alive(self):
        return self._alive

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


class RealOutputPanel:
    """真机输出状态面板: 控制 enable, 显示 sim 目标与真机编码器误差。"""

    def __init__(self, names, on_sit):
        import tkinter as tk
        self.tk = tk
        self.on_sit = on_sit
        self.root = tk.Tk()
        self.root.title("OceanBDX sim2real 输出")

        self.enabled = tk.BooleanVar(value=False)
        hdr = tk.Frame(self.root)
        hdr.pack(fill="x", padx=6, pady=4)
        tk.Checkbutton(hdr, text="使能电机输出", variable=self.enabled,
                       fg="red").pack(side="left")
        tk.Button(hdr, text="真机缓慢到蹲姿", command=self.on_sit).pack(side="left", padx=8)
        self.status = tk.Label(hdr, text="DISABLED", width=18, anchor="w")
        self.status.pack(side="left")

        self.rows = []
        for name in names:
            row = tk.Frame(self.root)
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=name, width=12, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="sim: --  real: --  err: --", width=44, anchor="w")
            lbl.pack(side="left")
            self.rows.append(lbl)

        self._alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        self._alive = False

    def is_enabled(self):
        return bool(self.enabled.get())

    def set_enabled(self, on):
        self.enabled.set(bool(on))

    def set_state(self, sim_q, real_q, tau=None, align_active=False, align_done=False):
        if align_active:
            text = "SIT ALIGN" if self.is_enabled() else "ALIGN OFF"
        elif align_done:
            text = "SIT READY"
        else:
            text = "ENABLED" if self.is_enabled() else "DISABLED"
        self.status.config(text=text)
        for i, lbl in enumerate(self.rows):
            t = "" if tau is None else f" tau:{tau[i]:+.1f}"
            lbl.config(text=(f"sim:{sim_q[i]:+.3f}  real:{real_q[i]:+.3f}  "
                             f"err:{(sim_q[i] - real_q[i]):+.3f}{t}"))

    def update(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self._alive = False

    def alive(self):
        return self._alive

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


class Policy:
    """路线 B（BDX 论文复刻）有界动作与 paper-aligned 观测构造。

    须与训练侧 oceanisaaclab .../oceanisaaclab_walk_env.py `_get_observations` 逐位一致：
      [0:2]   pos_pf × pos_pf_scale                path 系躯干 xy
      [2:4]   (sin, cos) 相对 yaw = head_yaw − path_yaw
      [4:7]   projected gravity                    body 系重力方向
      [7:10]  lin_vel_b × lin_vel_scale            body 系线速度
      [10:13] ang_vel_b × ang_vel_scale            body 系角速度（= gyro）
      [13:23] (q_leg − default) × dof_pos_scale
      [23:27] q_neck × dof_pos_scale               脖子 4 关节角（无 default 偏移）
      [27:37] dq_leg × dof_vel_scale
      [37:41] dq_neck × dof_vel_scale
      [41:55] a_{t-1}（14：10 腿 + 4 脖子）
      [55:69] a_{t-2}
      [69:73] (sin2πφ, cos2πφ, sin4πφ, cos4πφ)     相位二阶谐波
      [73:76] cmd × commands_scale
      [76:80] head_cmd × head_command_scale        (Δh, pitch, yaw, roll)
    动作解码（14 维）：前 10 腿 target = default + action_joint_ranges⊙clip；
      后 4 脖子 target = neck_default + neck_action_joint_ranges⊙clip。
    path frame 状态机（PathFrameNP）由外部用 MuJoCo 真值逐控制步 step() 推进。
    """

    def __init__(self, onnx_path, cfg, num_joints, num_neck=4, stand=False):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        expected_obs = 77 if stand else 80
        actual_obs = self.sess.get_inputs()[0].shape[-1]
        if actual_obs != expected_obs:
            raise RuntimeError(
                f"{onnx_path} expects {actual_obs} observations, but "
                f"{'stand' if stand else 'walk'} requires {expected_obs}. "
                "Retrain and export a compatible policy."
            )
        self.cfg = cfg
        self.nj = num_joints  # 腿关节数（10）
        self.n_neck = num_neck  # 脖子关节数（4）
        self.n_act = num_joints + num_neck  # 动作维度（14）
        # stand=True：独立训练的站立（perpetual）模型。观测去相位谐波、命令由
        #   (cmd3 行走速度) 换成 (torso4 躯干位姿命令 h,pitch,yaw,roll)，共 77 维；
        #   动作解码与行走完全一致。stand=False：行走模型，80 维（含姿态、phase4 + cmd3）。
        self.stand = bool(stand)
        self.default_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float32)
        self.neck_default_dof_pos = np.array(
            cfg.get("neck_default_dof_pos", [0.0] * num_neck), dtype=np.float32
        )
        self.commands_scale = np.array(cfg.get("commands_scale", [2.0, 2.0, 1.0]), dtype=np.float32)
        # 站立命令缩放 (h_torso, pitch, yaw, roll)；h 量级小，放大到与角度可比。
        self.torso_command_scale = np.array(
            cfg.get("torso_command_scale", [10.0, 1.0, 1.0, 1.0]), dtype=np.float32
        )
        self.head_command_scale = np.array(
            cfg.get("head_command_scale", [20.0, 1.0, 1.0, 1.0]), dtype=np.float32
        )
        self.gait_cycle_period = float(cfg.get("gait_cycle_period", 0.6))
        self.gait_period_fast = float(cfg.get("gait_period_fast", 0.48))
        self.policy_dt = float(cfg.get("policy_dt", 0.05))
        self.move_command_threshold = float(cfg.get("move_command_threshold", 0.08))
        # 路线 B 新增缩放/映射
        self.pos_pf_scale = float(cfg.get("pos_pf_scale", 4.0))
        self.lin_vel_scale = float(cfg.get("lin_vel_scale", 2.0))
        self.action_joint_ranges = np.array(
            cfg.get("action_joint_ranges", [0.35, 0.35, 0.8, 0.9, 0.8] * 2), dtype=np.float32
        )[:self.nj]
        self.neck_action_joint_ranges = np.array(
            cfg.get("neck_action_joint_ranges", [0.8, 0.8, 1.2, 0.7]), dtype=np.float32
        )[:self.n_neck]
        self.gait_phase = 0.0
        self.last_actions = np.zeros(self.n_act, dtype=np.float32)
        self.last_last_actions = np.zeros(self.n_act, dtype=np.float32)
        self.last_obs = None
        self.last_projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.last_raw_actions = np.zeros(self.n_act, dtype=np.float32)

    def reset(self, previous_actions=None, previous_previous_actions=None):
        """Reset recurrent policy-side state while optionally preserving action history.

        The paper relies on both policies receiving the same two physical actions across a
        transition.  Since stand and walk share the 14-D bounded action mapping, their
        normalized action histories can be transferred directly.
        """
        if previous_actions is None:
            self.last_actions[:] = 0
        else:
            previous_actions = np.asarray(previous_actions, dtype=np.float32)
            if previous_actions.shape != (self.n_act,):
                raise ValueError(f"previous_actions must have shape {(self.n_act,)}, got {previous_actions.shape}")
            self.last_actions[:] = np.clip(previous_actions, -self.cfg["clip_actions"], self.cfg["clip_actions"])
        if previous_previous_actions is None:
            self.last_last_actions[:] = self.last_actions
        else:
            previous_previous_actions = np.asarray(previous_previous_actions, dtype=np.float32)
            if previous_previous_actions.shape != (self.n_act,):
                raise ValueError(
                    f"previous_previous_actions must have shape {(self.n_act,)}, "
                    f"got {previous_previous_actions.shape}"
                )
            self.last_last_actions[:] = np.clip(
                previous_previous_actions, -self.cfg["clip_actions"], self.cfg["clip_actions"]
            )
        self.last_obs = None
        self.last_projected_gravity[:] = [0.0, 0.0, -1.0]
        self.last_raw_actions[:] = self.last_actions
        self.gait_phase = 0.0

    def step(self, q, dq, gyro, projected_gravity, cmd, pos_pf, yaw_pf, lin_vel_b,
             neck_q, neck_dq, head_cmd, torso_cmd=None):
        """一次策略推理（行走 80 维 / 站立 77 维，由 self.stand 决定观测尾部）。

        Args:
            q, dq: (nj,) 腿关节角/角速度（URDF 系）。
            gyro: (3,) body 系角速度。
            cmd: (3,) 行走头部系速度命令 (vx 前, vy 左, wz)；站立模型不使用。
            pos_pf: (2,) 躯干 path 系 xy（PathFrameNP.base_in_path_frame 提供）。
            yaw_pf: 相对 yaw = head_yaw − path_yaw。
            lin_vel_b: (3,) body 系线速度（真机来自状态估计器；sim2sim 用 MuJoCo 真值）。
            neck_q, neck_dq: (n_neck,) 脖子关节角/角速度。
            head_cmd: (4,) 头部命令 (Δh, pitch, yaw, roll)。
            torso_cmd: (4,) 站立躯干命令 (h, pitch, yaw, roll)；仅站立模型使用。

        Returns:
            (leg_target(nj,), neck_target(n_neck,))。
        """
        c = self.cfg
        cmd = np.asarray(cmd, dtype=np.float32)
        head_cmd = np.asarray(head_cmd, dtype=np.float32)
        yaw_feat = np.array([np.sin(yaw_pf), np.cos(yaw_pf)], dtype=np.float32)

        # 两个策略均使用 projected gravity 表达 torso roll/pitch。站立尾部少 phase4/cmd3，
        # 改为 torso_cmd4，因此为 77 维；行走为 80 维。
        obs_common = [
            np.asarray(pos_pf, dtype=np.float32) * self.pos_pf_scale,
            yaw_feat,
            np.asarray(projected_gravity, dtype=np.float32),
        ]
        obs_common += [
            np.asarray(lin_vel_b, dtype=np.float32) * self.lin_vel_scale,
            gyro * c["ang_vel_scale"],
            (q - self.default_dof_pos) * c["dof_pos_scale"],
            np.asarray(neck_q, dtype=np.float32) * c["dof_pos_scale"],
            dq * c["dof_vel_scale"],
            np.asarray(neck_dq, dtype=np.float32) * c["dof_vel_scale"],
            self.last_actions,
            self.last_last_actions,
        ]

        if self.stand:
            # 站立尾部：torso_cmd4 + head_cmd4（无相位谐波，共 77 维）。
            if torso_cmd is None:
                torso_cmd = np.zeros(4, dtype=np.float32)
            obs_tail = [
                np.asarray(torso_cmd, dtype=np.float32) * self.torso_command_scale,
                head_cmd * self.head_command_scale,
            ]
        else:
            # 行走尾部：phase 二阶谐波4 + cmd3 + head_cmd4（完整观测共 80 维）。
            # 相位积分：命令超过移动阈值时按速度改变步频；零速时冻结相位。
            # 训练侧 _pre_physics_step 在 _get_observations 之前推进相位，obs 见已推进的 φ。
            speed_fraction = min(
                1.0,
                max(
                    abs(float(cmd[0])) / max(1.0e-6, float(c.get("reference_vx_max", 0.25))),
                    abs(float(cmd[1])) / max(1.0e-6, float(c.get("reference_vy_max", 0.15))),
                    abs(float(cmd[2])) / max(1.0e-6, float(c.get("reference_wz_max", 0.8))),
                ),
            )
            period = self.gait_cycle_period + (self.gait_period_fast - self.gait_cycle_period) * speed_fraction
            moving = bool(np.max(np.abs(cmd)) > self.move_command_threshold)
            if moving:
                self.gait_phase = (self.gait_phase + self.policy_dt / period) % 1.0
            two_pi_phase = 2.0 * np.pi * self.gait_phase
            phase_feat = np.array([
                np.sin(two_pi_phase),
                np.cos(two_pi_phase),
                np.sin(2.0 * two_pi_phase),
                np.cos(2.0 * two_pi_phase),
            ], dtype=np.float32)
            obs_tail = [
                phase_feat,
                cmd * self.commands_scale,
                head_cmd * self.head_command_scale,
            ]

        obs = np.concatenate(obs_common + obs_tail).astype(np.float32)

        obs = np.clip(obs, -c["clip_obs"], c["clip_obs"])
        self.last_obs = obs.copy()
        act = self.sess.run(None, {self.input_name: obs[None, :]})[0][0]
        self.last_raw_actions = act.astype(np.float32)
        act = np.clip(act, -c["clip_actions"], c["clip_actions"])
        # 双帧动作历史滚动：先把上一帧移到 t-2，再写入本帧到 t-1
        self.last_last_actions = self.last_actions.copy()
        self.last_actions = act.astype(np.float32)
        # 逐关节线性映射：腿 0→标称站姿；脖子 0→默认位（±range）
        leg_target = self.default_dof_pos + self.action_joint_ranges * act[:self.nj]
        neck_target = self.neck_default_dof_pos + self.neck_action_joint_ranges * act[self.nj:]
        return leg_target, neck_target


class RunLogger:
    """把每个 policy step 的观测/动作/姿态写入带时间戳的 CSV (runlog/)。

    每行 = 一次 policy 推理 (RL_BALANCE / RL_WALK 下 decimation 对齐时)。
    观测拆成各物理量分列, 动作记录 raw(网络原始) / clip(裁剪后) / target(下发关节角)。
    """

    def __init__(self, nj, path=None):
        self.nj = nj
        os.makedirs(os.path.join(ROOT, "runlog"), exist_ok=True)
        if path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(ROOT, "runlog", f"sim2sim_{stamp}.csv")
        self.path = path
        self.f = open(path, "w", buffering=1)  # 行缓冲, 实时落盘
        leg = [j.replace("_joint", "") for j in LEG_JOINTS]
        actions = [j.replace("_joint", "") for j in ACTION_JOINTS]
        cols = (["t", "state", "cmd_vx", "cmd_vy", "cmd_wz",
                 "base_z", "tilt_x", "tilt_y",
                 "vel_bx", "vel_by", "vel_bz",
                 "gyro_x", "gyro_y", "gyro_z"]
                + [f"q_{j}" for j in leg]
                + [f"dq_{j}" for j in leg]
                + [f"raw_{j}" for j in actions]
                + [f"act_{j}" for j in actions]
                + [f"tgt_{j}" for j in leg])
        self.cols = cols
        self.f.write(",".join(cols) + "\n")
        print(f"[runlog] 写入 {path}")

    def log(self, t, state, cmd, base_z, tilt, vel_b, gyro, q, dq,
            raw_action, action, target):
        row = ([f"{t:.4f}", state,
                f"{cmd[0]:.4f}", f"{cmd[1]:.4f}", f"{cmd[2]:.4f}",
                f"{base_z:.4f}", f"{tilt[0]:.4f}", f"{tilt[1]:.4f}",
                f"{vel_b[0]:.4f}", f"{vel_b[1]:.4f}", f"{vel_b[2]:.4f}",
                f"{gyro[0]:.4f}", f"{gyro[1]:.4f}", f"{gyro[2]:.4f}"]
               + [f"{v:.4f}" for v in q]
               + [f"{v:.4f}" for v in dq]
               + [f"{v:.4f}" for v in raw_action]
               + [f"{v:.4f}" for v in action]
               + [f"{v:.4f}" for v in target])
        self.f.write(",".join(row) + "\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class Sim:
    def __init__(self, args):
        with open(args.config, encoding="utf-8") as config_file:
            full = yaml.safe_load(config_file)["oceanbdx"]
        self.cfg = full
        self.nj = len(LEG_JOINTS)

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        # 关节索引 (qpos/qvel中, 跳过freejoint)
        self.q_adr = np.array([self.model.joint(j).qposadr[0] for j in LEG_JOINTS])
        self.v_adr = np.array([self.model.joint(j).dofadr[0] for j in LEG_JOINTS])
        self.neck_q_adr = np.array([self.model.joint(j).qposadr[0] for j in NECK_JOINTS])
        self.neck_v_adr = np.array([self.model.joint(j).dofadr[0] for j in NECK_JOINTS])

        ctrl = full["control"]
        self.kp = np.array(ctrl["rl_kp"][:self.nj])
        self.kd = np.array(ctrl["rl_kd"][:self.nj])
        sim2sim_ctrl = full.get("sim2sim", {})
        if "rl_kp" in sim2sim_ctrl:
            self.kp = np.array(sim2sim_ctrl["rl_kp"][:self.nj])
        if "rl_kd" in sim2sim_ctrl:
            self.kd = np.array(sim2sim_ctrl["rl_kd"][:self.nj])
        if getattr(args, "sim_rl_kd", None) is not None:
            self.kd = np.full(self.nj, float(args.sim_rl_kd))
        if getattr(args, "sim_rl_kd_list", None):
            values = [float(v) for v in args.sim_rl_kd_list.split(",")]
            if len(values) != self.nj:
                raise ValueError(f"--sim-rl-kd-list expects {self.nj} comma-separated values")
            self.kd = np.array(values, dtype=float)
        self.fixed_kp = np.array(ctrl["fixed_kp"][:self.nj])
        self.fixed_kd = np.array(ctrl["fixed_kd"][:self.nj])
        self.tau_limit = np.array(ctrl["torque_limits"][:self.nj])
        self.control_dt = ctrl["dt"]
        self.decimation = ctrl["decimation"]
        self.stand_duration = ctrl["stand_duration"]
        self.rl_gain_blend_duration = max(0.0, float(ctrl.get("rl_gain_blend_duration", 0.5)))
        self.damping_kd = ctrl["damping_kd"]

        self.joint_lower = np.array(ctrl["joint_lower"][:self.nj], dtype=float)
        self.joint_upper = np.array(ctrl["joint_upper"][:self.nj], dtype=float)
        # Training-side appendix-B Go1 actuator model.
        self.motor_tau_max = np.full(self.nj, 23.7)
        self.motor_qd_tau_max = np.full(self.nj, 10.6)
        self.motor_qd_max = np.full(self.nj, 28.8)
        self.motor_mu_s = np.full(self.nj, 0.15)
        self.motor_mu_d = np.full(self.nj, 0.016)
        self.motor_qd_s = 0.1
        cutoff = float(sim2sim_ctrl.get("action_lowpass_cutoff_hz", 37.5))
        self.target_lowpass_alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff * self.control_dt)
        self.move_command_threshold = float(
            full["policy"].get("move_command_threshold", 0.08)
        )
        # walk -> stand uses only signals available on the real robot: controller phase,
        # projected gravity/gyro, leg q/dq and the applied joint target. MuJoCo contact
        # forces and true base velocity remain diagnostics and are never switch gates.
        self.switch_decel_duration = max(
            self.control_dt, float(sim2sim_ctrl.get("switch_decel_duration_s", 0.6))
        )
        self.switch_min_moving_command = max(
            self.move_command_threshold + 1.0e-4,
            float(sim2sim_ctrl.get("switch_min_moving_command", 0.09)),
        )
        phase_windows = np.asarray(
            sim2sim_ctrl.get(
                "switch_double_support_phase_windows",
                [[0.02, 0.08], [0.52, 0.58]],
            ),
            dtype=float,
        )
        if phase_windows.ndim != 2 or phase_windows.shape[1] != 2:
            raise ValueError("sim2sim.switch_double_support_phase_windows must be Nx2")
        self.switch_double_support_phase_windows = np.mod(phase_windows, 1.0)
        self.switch_zero_hold_duration = max(
            0.0, float(sim2sim_ctrl.get("switch_zero_hold_duration_s", 0.4))
        )
        self.switch_final_decel_duration = max(
            self.control_dt,
            float(sim2sim_ctrl.get("switch_final_decel_duration_s", 0.15)),
        )
        self.switch_stable_confirm_duration = max(
            self.control_dt,
            float(sim2sim_ctrl.get("switch_stable_confirm_duration_s", 0.2)),
        )
        self.switch_total_timeout = max(
            self.switch_decel_duration
            + self.switch_final_decel_duration
            + self.switch_zero_hold_duration,
            float(sim2sim_ctrl.get("switch_total_timeout_s", 3.5)),
        )
        self.switch_proj_g_xy_max = float(
            sim2sim_ctrl.get("switch_projected_gravity_xy_max", 0.20)
        )
        self.switch_upright_projection_min = float(
            sim2sim_ctrl.get("switch_upright_projection_min", 0.98)
        )
        self.switch_gyro_xy_max = float(sim2sim_ctrl.get("switch_gyro_xy_max", 0.12))
        self.switch_gyro_z_max = float(sim2sim_ctrl.get("switch_gyro_z_max", 0.15))
        self.switch_joint_vel_rms_max = float(
            sim2sim_ctrl.get("switch_joint_vel_rms_max", 0.30)
        )
        self.switch_joint_vel_max = float(
            sim2sim_ctrl.get("switch_joint_vel_max", 0.80)
        )
        self.switch_joint_pos_error_max = float(
            sim2sim_ctrl.get("switch_joint_pos_error_max", 0.25)
        )
        self.switch_target_error_max = float(
            sim2sim_ctrl.get("switch_target_error_max", 0.50)
        )
        # stand -> walk remains on the standing policy until its torso command is neutral
        # and real-robot-available IMU/encoder signals have stayed stable.
        self.stand_to_walk_recenter_max_duration = max(
            0.0,
            float(sim2sim_ctrl.get("stand_to_walk_recenter_max_duration_s", 1.5)),
        )
        self.stand_to_walk_zero_hold_duration = max(
            0.0,
            float(sim2sim_ctrl.get("stand_to_walk_zero_hold_duration_s", 0.30)),
        )
        self.stand_to_walk_stable_confirm_duration = max(
            self.control_dt,
            float(sim2sim_ctrl.get("stand_to_walk_stable_confirm_duration_s", 0.20)),
        )
        minimum_stand_to_walk_timeout = (
            self.stand_to_walk_recenter_max_duration
            + self.stand_to_walk_zero_hold_duration
            + self.stand_to_walk_stable_confirm_duration
        )
        self.stand_to_walk_total_timeout = max(
            minimum_stand_to_walk_timeout,
            float(sim2sim_ctrl.get("stand_to_walk_total_timeout_s", 5.0)),
        )
        self.policy_target_prev = np.zeros(self.nj)
        self.policy_target_next = np.zeros(self.nj)
        self.filtered_policy_target = np.zeros(self.nj)
        self.policy_substep = self.decimation

        self.neck_lower = np.array([self.model.joint(j).range[0] for j in NECK_JOINTS])
        self.neck_upper = np.array([self.model.joint(j).range[1] for j in NECK_JOINTS])
        self.neck_tau_limit = np.array(
            sim2sim_ctrl.get("neck_torque_limits", [5.0] * len(NECK_JOINTS)), dtype=float
        )

        cal = full["calibration"]
        self.sit_pose = np.array(cal["sit_pose"][:self.nj], dtype=float)
        self.stand_pose = np.array(cal["stand_pose"][:self.nj], dtype=float)

        cmd_cfg = full["command"]
        self.max_vel = np.array([cmd_cfg["max_vx"], cmd_cfg["max_vy"], cmd_cfg["max_wz"]])
        self.walk_command_accel_limits = np.asarray(
            cmd_cfg.get("walk_command_accel_limits", [0.50, 0.40, 1.50]),
            dtype=float,
        )
        self.walk_command_decel_limits = np.asarray(
            cmd_cfg.get("walk_command_decel_limits", [0.40, 0.30, 1.20]),
            dtype=float,
        )
        for name, limits in (
            ("walk_command_accel_limits", self.walk_command_accel_limits),
            ("walk_command_decel_limits", self.walk_command_decel_limits),
        ):
            if limits.shape != (3,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
                raise ValueError(f"command.{name} must contain three positive finite values")

        # 真机联调 (--real / --manual)
        self.real_cfg = full.get("real", {})
        self.want_real = getattr(args, "real", False)
        self.want_manual = getattr(args, "manual", False)
        self.bridge = None
        self.cmd_q = self.stand_pose.copy()
        self.real_panel = None
        self.real_output_enabled = False
        self.real_cmd_q = self.sit_pose.copy()
        self.real_sit_active = False
        self.real_sit_done = False
        self.real_sit_last_print = 0.0
        self.real_sit_ready_tol = float(self.real_cfg.get("sit_ready_tolerance", 0.12))
        self.real_sit_cmd_tol = float(self.real_cfg.get("sit_cmd_tolerance", 0.02))
        self.real_follow_kp = np.array(self.real_cfg.get("follow_kp", self.fixed_kp))[:self.nj]
        self.real_follow_kd = np.array(self.real_cfg.get("follow_kd", self.fixed_kd))[:self.nj]
        self.last_target = self.sit_pose.copy()
        self.last_kp = self.fixed_kp.copy()
        self.last_kd = self.fixed_kd.copy()

        # 两个独立模型：行走 (RL_WALK, 80 维) 与站立 (RL_STAND, 77 维)。
        # self.walk_policy / self.stand_policy 分别加载；self.policy 始终指向"当前
        # 活动模型"(在 control_step 里按状态切换)，供调试/日志/reset 复用。
        self.walk_policy = None
        self.stand_policy = None
        if not args.no_policy:
            pcfg = dict(full["policy"])
            pcfg["default_dof_pos"] = pcfg["default_dof_pos"][:self.nj]
            pcfg["policy_dt"] = float(full["control"]["dt"]) * int(full["control"]["decimation"])
            sim2sim_ctrl = full.get("sim2sim", {})
            if "action_scale" in sim2sim_ctrl:
                pcfg["action_scale"] = sim2sim_ctrl["action_scale"]
            if getattr(args, "sim_action_scale", None) is not None:
                pcfg["action_scale"] = float(args.sim_action_scale)

            walk_path = args.policy or os.path.join(ROOT, full["policy"]["path"])
            if os.path.exists(walk_path):
                self.walk_policy = Policy(walk_path, pcfg, self.nj,
                                          num_neck=len(NECK_JOINTS), stand=False)
                print(f"[sim2sim] walk policy loaded: {walk_path}")
            else:
                print(f"[sim2sim] walk policy not found at {walk_path}, RL_WALK unavailable")

            stand_rel = full["policy"].get("stand_path", "policy/stand/policy.onnx")
            stand_path = args.stand_policy or os.path.join(ROOT, stand_rel)
            if os.path.exists(stand_path):
                self.stand_policy = Policy(stand_path, pcfg, self.nj,
                                           num_neck=len(NECK_JOINTS), stand=True)
                print(f"[sim2sim] stand policy loaded: {stand_path}")
            else:
                print(f"[sim2sim] stand policy not found at {stand_path}, "
                      "RL_STAND 将回退到行走模型(零命令)")
        # self.policy: 当前活动模型引用（默认行走，probe/push 等调试路径沿用）
        self.policy = self.walk_policy

        # ---- 路线 B path frame（论文 V-A）：MuJoCo 真值驱动 ----
        pcfg_full = full["policy"]
        # head_yaw offset：forward_vx_sign=-1 → offset=π（URDF base +x 指尾部，头前向=−x）
        self.forward_vx_sign = float(pcfg_full.get("forward_vx_sign", -1.0))
        self.head_yaw_offset = 0.0 if self.forward_vx_sign > 0.0 else np.pi
        self.move_command_threshold = float(pcfg_full.get("move_command_threshold", 0.08))
        self.path_frame = PathFrameNP(
            stand_time_constant=float(pcfg_full.get("path_frame_stand_time_constant", 1.0)),
            max_pos_deviation=float(pcfg_full.get("path_frame_max_pos_deviation", 0.25)),
            max_yaw_deviation=float(pcfg_full.get("path_frame_max_yaw_deviation", 0.6)),
        )
        # Contact uses sole geoms, while path-frame kinematics must use the terminal link
        # origins/quaternions, matching IsaacLab's body_pos_w/body_quat_w exactly.
        self.foot_geom_ids = [self.model.geom("foot_r").id, self.model.geom("foot_l").id]
        self.foot_body_ids = [int(self.model.geom_bodyid[gid]) for gid in self.foot_geom_ids]
        self.floor_geom_id = self.model.geom("floor").id
        self.ground_geom_ids = frozenset(
            self.model.geom(name).id for name in GROUND_GEOM_NAMES
        )
        self.ground_geom_group = np.zeros(6, dtype=np.uint8)
        for geom_id in self.ground_geom_ids:
            self.ground_geom_group[self.model.geom_group[geom_id]] = 1
        # 左右脚链是镜像的，两个 foot frame 的局部 +x 在 neutral pose 中相反。
        # 从 q=0 模型姿态标定各自到“头部前向”的 yaw offset，运行时再做圆均值。
        mujoco.mj_forward(self.model, self.data)
        neutral_head_yaw = yaw_from_quat(self.data.qpos[3:7]) + self.head_yaw_offset
        neutral_foot_yaws = np.array([
            np.arctan2(
                self.data.xmat[bid].reshape(3, 3)[1, 0],
                self.data.xmat[bid].reshape(3, 3)[0, 0],
            )
            for bid in self.foot_body_ids
        ])
        self.foot_heading_offsets = np.array(
            [wrap_angle(neutral_head_yaw - yaw) for yaw in neutral_foot_yaws], dtype=float
        )
        neutral_feet_center = np.mean(
            [self.data.xpos[bid, :2] for bid in self.foot_body_ids], axis=0
        )
        neutral_rel = self.data.qpos[:2] - neutral_feet_center
        cos_y, sin_y = np.cos(neutral_head_yaw), np.sin(neutral_head_yaw)
        self.neutral_torso_pos_pf = np.array([
            neutral_rel[0] * cos_y + neutral_rel[1] * sin_y,
            -neutral_rel[0] * sin_y + neutral_rel[1] * cos_y,
        ], dtype=np.float32)

        # ---- 2026-07-08 脖子/头部命令 ----
        self.n_neck = len(NECK_JOINTS)
        self.head_cmd = np.zeros(4, dtype=np.float32)  # (Δh, pitch, yaw, roll)
        # 站立躯干命令 (h, pitch, yaw, roll)，仅 RL_STAND 生效；默认全 0=标称直立站姿。
        # 键盘微调见 _cmd_key：t/g=pitch± v/c=yaw± y/b=roll± f/z=高度±（切状态清零）。
        # torso_cmd is the user target. _effective_torso_cmd also carries the scripted
        # STAND_UP hand-off and the standing-policy recenter before a walk switch.
        self.torso_cmd = np.zeros(4, dtype=np.float32)
        self._effective_torso_cmd = np.zeros(4, dtype=np.float32)
        cmd_cfg = full.get("command", {})
        # 站立高度命令是不对称范围，必须显式保存 min/max。
        self.torso_command_min = np.array(
            cmd_cfg.get("torso_command_min", [-0.04, -0.17, -0.24, -0.09]), dtype=np.float32
        )
        self.torso_command_max = np.array(
            cmd_cfg.get("torso_command_max", [0.01, 0.17, 0.24, 0.09]), dtype=np.float32
        )
        if self.torso_command_min.shape != (4,) or self.torso_command_max.shape != (4,):
            raise ValueError("command.torso_command_min/max must each contain four values")
        if np.any(self.torso_command_min >= self.torso_command_max):
            raise ValueError("command.torso_command_min must be smaller than torso_command_max")
        self.stand_base_height = float(pcfg_full.get("stand_base_height", 0.38498640060424805))
        self.walk_max_head = np.array([
            float(cmd_cfg.get("max_head_dh", 0.007)),
            float(cmd_cfg.get("max_head_pitch", 0.17)),
            float(cmd_cfg.get("max_head_yaw", 0.33)),
            float(cmd_cfg.get("max_head_roll", 0.20)),
        ], dtype=np.float32)
        self.stand_max_head = np.array([
            float(cmd_cfg.get("stand_max_head_dh", 0.02)),
            float(cmd_cfg.get("stand_max_head_pitch", 0.50)),
            float(cmd_cfg.get("stand_max_head_yaw", 1.00)),
            float(cmd_cfg.get("stand_max_head_roll", 0.60)),
        ], dtype=np.float32)
        self.puppeteering_cfg = full.get("puppeteering", {})
        self.gamepad_enabled = bool(getattr(args, "gamepad", False))
        self.gamepad = None
        self.puppeteer = PuppeteeringMapper(
            self.puppeteering_cfg,
            self.max_vel,
            self.torso_command_min,
            self.torso_command_max,
            self.stand_max_head,
            walking_max_head=self.walk_max_head,
        )
        self._puppeteer_walk_requested = False
        self._puppeteer_connected = False
        self._puppeteer_full_speed = False
        self._puppeteer_explicit_stand_request = False
        self._puppeteer_blocked_target = None
        self._stand_ready_elapsed = 0.0
        self.neck_kp = float(pcfg_full.get("neck_kp", 50.0))
        self.neck_kd = float(pcfg_full.get("neck_kd", 2.0))
        neck_default = pcfg_full.get("neck_default_dof_pos", [0.0] * self.n_neck)
        self.neck_default = np.array(neck_default, dtype=np.float32)
        # 脖子位置目标（策略输出，RL 前锁默认位）
        self.neck_target = self.neck_default.copy()
        self.neck_policy_target_prev = self.neck_default.copy()
        self.neck_policy_target_next = self.neck_default.copy()
        self.filtered_neck_target = self.neck_default.copy()

        self.state = "SIT"
        self.state_time = 0.0
        self._rl_switch_requests = queue.SimpleQueue()
        self.pending_rl_state = None
        self._effective_walk_cmd = np.zeros(3, dtype=float)
        self._walk_stop_stage = None
        self._walk_stop_start_cmd = np.zeros(3, dtype=float)
        self._walk_stop_total_elapsed = 0.0
        self._walk_stop_stage_elapsed = 0.0
        self._walk_stop_stable_elapsed = 0.0
        self._walk_stop_block_reason = "idle"
        self._walk_stop_source = None
        self._stand_to_walk_stage = None
        self._stand_to_walk_total_elapsed = 0.0
        self._stand_to_walk_stage_elapsed = 0.0
        self._stand_to_walk_stable_elapsed = 0.0
        self._stand_to_walk_recenter_duration = 0.0
        self._stand_to_walk_start_torso_cmd = np.zeros(4, dtype=np.float32)
        self._stand_to_walk_start_head_cmd = np.zeros(4, dtype=np.float32)
        self._stand_to_walk_effective_head_cmd = np.zeros(4, dtype=np.float32)
        self._stand_to_walk_block_reason = "idle"
        self._stand_to_walk_cancelled = False
        self._stand_to_walk_source = None
        self._stand_to_walk_preserve_walk_command = False
        self._stand_to_walk_preconfirmed_stable = False
        self._walk_command_safe_zero_active = False
        self._rl_gain_blend_active = False
        self._rl_gain_blend_elapsed = 0.0
        self._rl_gain_blend_ratio = 1.0
        self._rl_gain_blend_start_target = self.stand_pose.copy()
        self._rl_gain_blend_start_neck_target = self.neck_default.copy()
        self._rl_gain_blend_start_torso_command = np.zeros(4, dtype=np.float32)
        self._rl_gain_blend_target_torso_command = np.zeros(4, dtype=np.float32)
        self.cmd = np.zeros(3)
        self.run_logger = None
        self.want_runlog = not bool(getattr(args, "no_log", False))
        self.rl_target = self.stand_pose.copy()
        self.policy_target_prev = self.stand_pose.copy()
        self.policy_target_next = self.stand_pose.copy()
        self.filtered_policy_target = self.stand_pose.copy()
        self.policy_substep = self.decimation
        self.rl_tick = 0
        self.stand_start = self.sit_pose.copy()
        self.debug_obs = bool(getattr(args, "debug_obs", False))
        self.debug_actions = bool(getattr(args, "debug_actions", False))
        self.debug_actions_full = bool(getattr(args, "debug_actions_full", False))
        self.debug_balance = bool(getattr(args, "debug_balance", False))
        self.debug_obs_interval = float(getattr(args, "debug_obs_interval", 0.2))
        self._last_debug_obs_time = -1.0e9
        self.debug_push_steps = int(getattr(args, "debug_push_steps", 0))
        self.debug_push_start = int(getattr(args, "debug_push_start", 80))
        self.debug_push_duration = int(getattr(args, "debug_push_duration", 11))
        self.debug_push_force = np.array([
            float(getattr(args, "debug_push_force_x", 0.0)),
            float(getattr(args, "debug_push_force_y", 40.0)),
            0.0,
        ], dtype=float)
        self.debug_push_policy_step = 0
        self.debug_push_base_body_id = self.model.body("base_link").id
        self.debug_push_foot_geom_ids = [self.model.geom("foot_r").id, self.model.geom("foot_l").id]
        self.viewer_push_force = np.array([
            float(getattr(args, "viewer_push_force_x", 0.0)),
            float(getattr(args, "viewer_push_force_y", 0.0)),
            float(getattr(args, "viewer_push_force_z", 0.0)),
        ], dtype=float)
        self.viewer_push_duration = float(getattr(args, "viewer_push_duration", 0.1))
        if not np.all(np.isfinite(self.viewer_push_force)):
            raise ValueError("viewer push force must contain only finite values")
        if not np.isfinite(self.viewer_push_duration) or self.viewer_push_duration <= 0.0:
            raise ValueError("viewer push duration must be finite and greater than zero")
        self._viewer_push_remaining = 0.0
        self._viewer_push_sign = 1.0
        self._viewer_push_requests = queue.SimpleQueue()
        self.viewer_push_base_body_id = self.model.body("base_link").id

        # 虚拟坐凳 (复刻真机底座): 坐姿时支撑躯干, 起立完成后下沉移除
        self.stool_gid = self.model.geom("stool").id
        self.stool_pos0 = self.model.geom_pos[self.stool_gid].copy()
        # 坐姿底座高度 = 凳子顶面 + base底部到原点距离 (0.1853, 由mesh测得)
        stool_top = self.stool_pos0[2] + self.model.geom_size[self.stool_gid][2]
        self.sit_base_height = stool_top + 0.1853 + 0.001  # 1mm 余量

        self.reset()

    def probe_policy(self):
        if not self.policy:
            print("[probe_policy] policy is not loaded")
            return
        q = self.stand_pose.copy()
        dq = np.zeros(self.nj, dtype=np.float32)
        gyro = np.zeros(3, dtype=np.float32)
        cmd = np.zeros(3, dtype=np.float32)
        # 理想直立零命令：path frame 已收敛到双脚中心，保留真实 torso-foot 偏置。
        pos_pf = self.neutral_torso_pos_pf.copy()
        yaw_pf = 0.0
        lin_vel_b = np.zeros(3, dtype=np.float32)
        neck_q = np.zeros(self.n_neck, dtype=np.float32)
        neck_dq = np.zeros(self.n_neck, dtype=np.float32)
        head_cmd = np.zeros(4, dtype=np.float32)
        target, _neck_target = self.policy.step(
            q, dq, gyro, np.array([0.0, 0.0, -1.0], dtype=np.float32),
            cmd, pos_pf, yaw_pf, lin_vel_b, neck_q, neck_dq, head_cmd
        )
        action = self.policy.last_actions
        raw_action = self.policy.last_raw_actions
        saturated = [ACTION_JOINTS[i] for i, value in enumerate(np.abs(action)) if value > 0.98]
        active = np.argsort(-np.abs(action))[:4]
        active_text = ", ".join(f"{ACTION_JOINTS[i]}={action[i]:+.3f}" for i in active)
        saturated_text = ",".join(saturated) if saturated else "none"
        print(
            "[probe_policy] ideal_zero_stand "
            f"act_absmax={np.max(np.abs(action)):.3f} "
            f"raw_act_absmax={np.max(np.abs(raw_action)):.3f} "
            f"target_absmax={np.max(np.abs(target)):.3f} "
            f"sat_count={len(saturated)}/{self.nj}"
        )
        print(f"[probe_policy] top=[{active_text}] saturated=[{saturated_text}]")
        print(f"[probe_policy] raw_action={np.array2string(raw_action, precision=3, floatmode='fixed', separator=', ')}")
        print(f"[probe_policy] target={np.array2string(target, precision=3, floatmode='fixed', separator=', ')}")

    def _remove_stool(self):
        """起立完成后将凳子沉入地下, 避免行走时绊脚"""
        if self.model.geom_pos[self.stool_gid][2] > -0.5:
            self.model.geom_pos[self.stool_gid][2] = -1.0
            print("[sim2sim] stool removed")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model.geom_pos[self.stool_gid] = self.stool_pos0  # 恢复凳子
        # 坐姿初始化: 躯干坐在凳子上, 双脚着地 (与真机底座启动一致)
        self.data.qpos[self.q_adr] = self.sit_pose
        self.data.qpos[2] = self.sit_base_height
        mujoco.mj_forward(self.model, self.data)
        self.state = "SIT"
        self.state_time = 0.0
        self._rl_switch_requests = queue.SimpleQueue()
        self.pending_rl_state = None
        self._effective_walk_cmd[:] = 0.0
        self._walk_stop_stage = None
        self._walk_stop_start_cmd[:] = 0.0
        self._walk_stop_total_elapsed = 0.0
        self._walk_stop_stage_elapsed = 0.0
        self._walk_stop_stable_elapsed = 0.0
        self._walk_stop_block_reason = "idle"
        self._walk_stop_source = None
        self._stand_to_walk_stage = None
        self._stand_to_walk_total_elapsed = 0.0
        self._stand_to_walk_stage_elapsed = 0.0
        self._stand_to_walk_stable_elapsed = 0.0
        self._stand_to_walk_recenter_duration = 0.0
        self._stand_to_walk_start_torso_cmd[:] = 0.0
        self._stand_to_walk_start_head_cmd[:] = 0.0
        self._stand_to_walk_effective_head_cmd[:] = 0.0
        self._stand_to_walk_block_reason = "idle"
        self._stand_to_walk_cancelled = False
        self._stand_to_walk_source = None
        self._stand_to_walk_preserve_walk_command = False
        self._stand_to_walk_preconfirmed_stable = False
        self._walk_command_safe_zero_active = False
        self.puppeteer.reset()
        self._puppeteer_walk_requested = False
        self._puppeteer_connected = False
        self._puppeteer_full_speed = False
        self._puppeteer_explicit_stand_request = False
        self._puppeteer_blocked_target = None
        self._stand_ready_elapsed = 0.0
        self._rl_gain_blend_active = False
        self._rl_gain_blend_elapsed = 0.0
        self._rl_gain_blend_ratio = 1.0
        self._rl_gain_blend_start_target = self.stand_pose.copy()
        self._rl_gain_blend_start_neck_target = self.neck_default.copy()
        self._rl_gain_blend_start_torso_command[:] = 0.0
        self._rl_gain_blend_target_torso_command[:] = 0.0
        self._viewer_push_remaining = 0.0
        self._viewer_push_requests = queue.SimpleQueue()
        self.cmd[:] = 0
        self.rl_target = self.stand_pose.copy()
        self.policy_target_prev = self.stand_pose.copy()
        self.policy_target_next = self.stand_pose.copy()
        self.filtered_policy_target = self.stand_pose.copy()
        self.policy_substep = self.decimation
        self.rl_tick = 0
        self.last_target = self.sit_pose.copy()
        self.last_kp = self.fixed_kp.copy()
        self.last_kd = self.fixed_kd.copy()
        self.neck_target = self.neck_default.copy()
        self.neck_policy_target_prev = self.neck_default.copy()
        self.neck_policy_target_next = self.neck_default.copy()
        self.filtered_neck_target = self.neck_default.copy()
        self.head_cmd[:] = 0.0
        self.torso_cmd[:] = 0.0
        self._effective_torso_cmd[:] = 0.0
        for pol in (self.walk_policy, self.stand_policy):
            if pol is not None:
                pol.reset()
        self.policy = self.walk_policy  # 活动模型引用复位到行走（默认）
        print("[sim2sim] reset to SIT")

    # ---------- 状态 ----------
    def get_obs_raw(self):
        q = self.data.qpos[self.q_adr].copy()
        dq = self.data.qvel[self.v_adr].copy()
        quat = self.data.qpos[3:7].copy()  # freejoint四元数 (w,x,y,z)
        gyro = self.data.qvel[3:6].copy()  # MuJoCo freejoint qvel[3:6] 即机体系角速度
        return q, dq, quat, gyro

    def debug_print_obs(self, q, dq, quat, gyro, target):
        if not self.debug_obs:
            return
        now = self.data.time
        if now - self._last_debug_obs_time < self.debug_obs_interval:
            return
        self._last_debug_obs_time = now

        projected_gravity = quat_rotate_inverse_gravity(quat)
        if self.policy:
            raw_action = self.policy.last_raw_actions
            action = self.policy.last_actions
            obs = self.policy.last_obs
            obs_min = float(np.min(obs)) if obs is not None else 0.0
            obs_max = float(np.max(obs)) if obs is not None else 0.0
        else:
            raw_action = np.zeros(self.n_neck + self.nj)
            action = np.zeros(self.n_neck + self.nj)
            obs_min = 0.0
            obs_max = 0.0

        print(
            "[debug_obs] "
            f"t={now:7.3f} state={self.state:10s} "
            f"q_abs={np.max(np.abs(q)):.3f} dq_abs={np.max(np.abs(dq)):.3f} "
            f"act_absmax={np.max(np.abs(action)):.3f} raw_act_absmax={np.max(np.abs(raw_action)):.3f} "
            f"target_absmax={np.max(np.abs(target)):.3f} obs_range=[{obs_min:+.3f},{obs_max:+.3f}]"
        )

        if self.debug_balance:
            left_force, right_force = self.foot_contact_forces()
            lin_vel_w, lin_vel_b = self.base_linear_velocities()
            print(
                "[debug_balance] "
                f"base_xy=[{self.data.qpos[0]:+.3f},{self.data.qpos[1]:+.3f}] "
                f"base_z={self.data.qpos[2]:+.3f} "
                f"vel_b=[{lin_vel_b[0]:+.3f},{lin_vel_b[1]:+.3f},{lin_vel_b[2]:+.3f}] "
                f"vel_w=[{lin_vel_w[0]:+.3f},{lin_vel_w[1]:+.3f},{lin_vel_w[2]:+.3f}] "
                f"tilt_xy=[{projected_gravity[0]:+.3f},{projected_gravity[1]:+.3f}] "
                f"feet_fz=[L:{left_force:.1f},R:{right_force:.1f}] "
                f"feet_balance={left_force - right_force:+.1f}"
            )

        if self.debug_actions:
            active = np.argsort(-np.abs(action))[:4]
            active_text = ", ".join(
                f"{ACTION_JOINTS[i]}={action[i]:+.3f}" for i in active
            )
            saturated = [ACTION_JOINTS[i] for i, value in enumerate(np.abs(action)) if value > 0.98]
            saturated_text = ",".join(saturated) if saturated else "none"
            print(
                "[debug_action_summary] "
                f"top=[{active_text}] saturated=[{saturated_text}] "
                f"sat_count={len(saturated)}/{self.nj}"
            )
            if self.debug_actions_full:
                print(
                    "[debug_action_detail] "
                    f"raw_action={np.array2string(raw_action, precision=3, floatmode='fixed', separator=', ')} "
                    f"q={np.array2string(q, precision=3, floatmode='fixed', separator=', ')} "
                    f"dq={np.array2string(dq, precision=3, floatmode='fixed', separator=', ')} "
                    f"action={np.array2string(action, precision=3, floatmode='fixed', separator=', ')} "
                    f"target={np.array2string(target, precision=3, floatmode='fixed', separator=', ')}"
                )

    def base_linear_velocities(self):
        lin_vel_w = self.data.qvel[0:3].copy()
        base_body_id = self.model.body("base_link").id
        rot_wb = self.data.xmat[base_body_id].reshape(3, 3)
        lin_vel_b = rot_wb.T @ lin_vel_w
        return lin_vel_w, lin_vel_b

    def path_frame_truth(self):
        """从 MuJoCo 读 path frame 所需真值：躯干、双脚中心/朝向与 body 线速度。

        真机侧这些量改由状态估计器/里程计 + FK 提供（论文 V-D 要求 runtime 一致）。
        """
        base_xy = self.data.qpos[0:2].copy()
        head_yaw = yaw_from_quat(self.data.qpos[3:7]) + self.head_yaw_offset
        feet_center = np.mean([self.data.xpos[bid, :2] for bid in self.foot_body_ids], axis=0)
        foot_yaws = np.array([
            np.arctan2(
                self.data.xmat[bid].reshape(3, 3)[1, 0],
                self.data.xmat[bid].reshape(3, 3)[0, 0],
            )
            for bid in self.foot_body_ids
        ])
        calibrated_yaws = foot_yaws + self.foot_heading_offsets
        feet_heading = np.arctan2(np.mean(np.sin(calibrated_yaws)), np.mean(np.cos(calibrated_yaws)))
        _, lin_vel_b = self.base_linear_velocities()
        return base_xy, head_yaw, feet_center, feet_heading, lin_vel_b

    def _stand_command_from_reliable_pose(self, target_command):
        """Build a hand-off command from signals that are reliable on the real robot.

        Roll and pitch come from the IMU quaternion. Height and path-relative yaw require a
        state estimator/FK and are therefore kept at the requested standing target instead of
        using MuJoCo-only truth. This keeps simulation and the future hardware FSM equivalent.
        """
        command = np.asarray(target_command, dtype=np.float32).copy()
        quat = self.data.qpos[3:7]
        roll, pitch, _ = rpy_from_quat(quat)
        command[1] = pitch
        command[3] = roll
        return np.clip(command, self.torso_command_min, self.torso_command_max)

    def foot_contact_forces(self):
        left_force = 0.0
        right_force = 0.0
        right_gid, left_gid = self.foot_geom_ids
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = (contact.geom1, contact.geom2)
            if self.ground_geom_ids.isdisjoint(pair):
                continue
            if left_gid not in pair and right_gid not in pair:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = max(0.0, float(force[0]))
            if contact.geom1 == left_gid or contact.geom2 == left_gid:
                left_force += normal_force
            if contact.geom1 == right_gid or contact.geom2 == right_gid:
                right_force += normal_force
        return left_force, right_force

    def ground_height_at(self, xy):
        """Return the highest terrain surface below a world-space XY point."""
        ray_z = max(2.0, float(self.data.qpos[2]) + 1.0)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            np.array([float(xy[0]), float(xy[1]), ray_z]),
            np.array([0.0, 0.0, -1.0]),
            self.ground_geom_group,
            1,
            -1,
            geom_id,
        )
        return ray_z - distance if distance >= 0.0 else float("nan")

    def _configure_viewer_camera(self, viewer):
        """Keep the small robot framed after the arena increases model extent."""
        base_body_id = self.model.body("base_link").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_body_id
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = self.data.xpos[base_body_id]
        viewer.cam.lookat[2] += 0.10
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

    def foot_geom_heights(self):
        return [float(self.data.geom_xpos[gid, 2]) for gid in self.debug_push_foot_geom_ids]

    def _print_debug_push(self, step, active, force_w, q, dq, target):
        lin_vel_w, lin_vel_b = self.base_linear_velocities()
        gravity_b = quat_rotate_inverse_gravity(self.data.qpos[3:7])
        left_force, right_force = self.foot_contact_forces()
        action = self.policy.last_actions if self.policy else np.zeros(self.nj)
        raw_action = self.policy.last_raw_actions if self.policy else np.zeros(self.nj)
        print(
            "[sim_push] "
            f"step={step} active={int(active)} "
            f"force_w=[{force_w[0]:+.1f},{force_w[1]:+.1f},{force_w[2]:+.1f}] "
            f"base_pos={np.round(self.data.qpos[0:3], 3).tolist()} "
            f"lin_vel_b={np.round(lin_vel_b, 3).tolist()} "
            f"lin_vel_w={np.round(lin_vel_w, 3).tolist()} "
            f"grav_b={np.round(gravity_b, 3).tolist()} "
            f"feet_fz=[R:{right_force:.1f},L:{left_force:.1f}] "
            f"foot_z={np.round(self.foot_geom_heights(), 3).tolist()} "
            f"q={np.round(q, 3).tolist()} "
            f"dq={np.round(dq, 3).tolist()} "
            f"action={np.round(action, 3).tolist()} "
            f"raw_action={np.round(raw_action, 3).tolist()} "
            f"target={np.round(target, 3).tolist()}"
        )

    def run_debug_push(self):
        if not self.policy:
            print("[sim_push] policy is not loaded; abort")
            return
        sim_steps_per_ctrl = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        print(
            "[sim_push] "
            f"force_w={self.debug_push_force.tolist()}N "
            f"start={self.debug_push_start} duration={self.debug_push_duration} "
            f"steps={self.debug_push_steps} sim_dt={self.model.opt.timestep} control_dt={self.control_dt} "
            f"decimation={self.decimation} action_scale={self.policy.cfg['action_scale']} kd={np.round(self.kd, 3).tolist()}"
        )
        self.stand_start = self.data.qpos[self.q_adr].copy()
        self.state, self.state_time = "STAND_UP", 0.0
        print("[sim_push] auto SIT -> STAND_UP")

        self.debug_push_policy_step = 0
        last_policy_tick = -1
        samples = []
        while self.debug_push_policy_step < self.debug_push_steps:
            tau, neck_tau = self.control_step()
            in_rl = self.state in ("RL_STAND", "RL_WALK")
            is_policy_tick = in_rl and self.policy and self.rl_tick != last_policy_tick and ((self.rl_tick - 1) % self.decimation == 0)
            active = in_rl and self.debug_push_start <= self.debug_push_policy_step < self.debug_push_start + self.debug_push_duration
            if is_policy_tick:
                last_policy_tick = self.rl_tick
                step = self.debug_push_policy_step
                force_w = self.debug_push_force if active else np.zeros(3)
                q, dq, _, _ = self.get_obs_raw()
                should_print = active or step % 10 == 0
                lin_vel_w, lin_vel_b = self.base_linear_velocities()
                left_force, right_force = self.foot_contact_forces()
                foot_z = self.foot_geom_heights()
                samples.append({
                    "step": step,
                    "active": active,
                    "base_x": float(self.data.qpos[0]),
                    "base_y": float(self.data.qpos[1]),
                    "base_z": float(self.data.qpos[2]),
                    "vel_x": float(lin_vel_b[0]),
                    "vel_y": float(lin_vel_b[1]),
                    "tilt_x": float(quat_rotate_inverse_gravity(self.data.qpos[3:7])[0]),
                    "tilt_y": float(quat_rotate_inverse_gravity(self.data.qpos[3:7])[1]),
                    "right_fz": float(right_force),
                    "left_fz": float(left_force),
                    "right_z": float(foot_z[0]),
                    "left_z": float(foot_z[1]),
                    "act_absmax": float(np.max(np.abs(self.policy.last_actions))),
                    "sat_count": int(np.sum(np.abs(self.policy.last_actions) > 0.98)),
                })
                if should_print:
                    self._print_debug_push(step, active, force_w, q, dq, self.rl_target)
                self.debug_push_policy_step += 1

            for _ in range(sim_steps_per_ctrl):
                self.data.qfrc_applied[:] = 0.0
                self.data.xfrc_applied[:] = 0.0
                self.data.qfrc_applied[self.v_adr] = tau
                self.data.qfrc_applied[self.neck_v_adr] = neck_tau
                if active:
                    self.data.xfrc_applied[self.debug_push_base_body_id, 0:3] = self.debug_push_force
                mujoco.mj_step(self.model, self.data)

        if samples:
            final = samples[-1]
            max_abs_vel_x = max(abs(s["vel_x"]) for s in samples)
            max_abs_vel_y = max(abs(s["vel_y"]) for s in samples)
            max_left_z = max(s["left_z"] for s in samples)
            max_right_z = max(s["right_z"] for s in samples)
            left_air_steps = sum(1 for s in samples if s["left_fz"] < 1.0)
            right_air_steps = sum(1 for s in samples if s["right_fz"] < 1.0)
            active_samples = [s for s in samples if s["active"]]
            peak_push_vel_x = max((abs(s["vel_x"]) for s in active_samples), default=0.0)
            peak_push_vel_y = max((abs(s["vel_y"]) for s in active_samples), default=0.0)
            print(
                "[sim_push_summary] "
                f"action_scale={self.policy.cfg['action_scale']:.3f} kd={np.round(self.kd, 3).tolist()} "
                f"final_step={final['step']} final_xy=[{final['base_x']:+.3f},{final['base_y']:+.3f}] "
                f"final_z={final['base_z']:+.3f} "
                f"final_vel_b=[{final['vel_x']:+.3f},{final['vel_y']:+.3f}] "
                f"final_tilt_xy=[{final['tilt_x']:+.3f},{final['tilt_y']:+.3f}] "
                f"max_abs_vel_b=[{max_abs_vel_x:.3f},{max_abs_vel_y:.3f}] "
                f"peak_push_abs_vel_b=[{peak_push_vel_x:.3f},{peak_push_vel_y:.3f}] "
                f"max_foot_z=[R:{max_right_z:.3f},L:{max_left_z:.3f}] "
                f"air_steps=[R:{right_air_steps},L:{left_air_steps}] "
                f"final_sat={final['sat_count']}/{self.nj}"
            )

    def _trigger_viewer_push(self, sign=1.0):
        """Start a repeatable viewer push on the main control thread."""
        if self.want_real:
            print("[viewer_push] disabled while --real is active")
            return False
        if self.state not in ("RL_STAND", "RL_WALK"):
            print(f"[viewer_push] ignored in {self.state}; enter RL_STAND or RL_WALK first")
            return False
        if np.linalg.norm(self.viewer_push_force) <= 1.0e-12:
            print("[viewer_push] force is zero; set --viewer-push-force-x/y/z")
            return False
        self._viewer_push_sign = 1.0 if sign >= 0.0 else -1.0
        self._viewer_push_remaining = self.viewer_push_duration
        force = self._viewer_push_sign * self.viewer_push_force
        print(
            "[viewer_push] start "
            f"state={self.state} force_w={np.round(force, 3).tolist()}N "
            f"duration={self.viewer_push_duration:.3f}s"
        )
        return True

    def _request_viewer_push(self, sign=1.0):
        """Queue a key-thread request; the control thread validates and starts it."""
        self._viewer_push_requests.put(1.0 if sign >= 0.0 else -1.0)

    def _consume_viewer_push_requests(self):
        """Drain queued key events at a control boundary, with the newest direction winning."""
        requested_sign = None
        while True:
            try:
                requested_sign = self._viewer_push_requests.get_nowait()
            except queue.Empty:
                break
        if requested_sign is not None:
            self._trigger_viewer_push(requested_sign)

    def _begin_viewer_push_substep(self):
        """Return the configured force for one physics substep, or cancel outside RL."""
        if self._viewer_push_remaining <= 0.0:
            return None
        if self.state not in ("RL_STAND", "RL_WALK"):
            self._viewer_push_remaining = 0.0
            print(f"[viewer_push] cancelled in {self.state}")
            return None
        return (self._viewer_push_sign * self.viewer_push_force).copy()

    def _apply_viewer_push_substep(self, applied_force):
        """Map a world-frame force at the base CoM into generalized force."""
        if applied_force is None:
            return
        mujoco.mj_applyFT(
            self.model,
            self.data,
            applied_force,
            np.zeros(3),
            self.data.xipos[self.viewer_push_base_body_id],
            self.viewer_push_base_body_id,
            self.data.qfrc_applied,
        )

    def _end_viewer_push_substep(self, applied_force):
        """Advance the scripted-force timer after one completed physics substep."""
        if applied_force is None:
            return
        was_active = self._viewer_push_remaining > 0.0
        self._viewer_push_remaining = max(
            0.0, self._viewer_push_remaining - float(self.model.opt.timestep)
        )
        if was_active and self._viewer_push_remaining <= 0.0:
            print("[viewer_push] complete")

    def key_cb(self, keycode):
        """MuJoCo viewer 窗口内的按键回调。

        ★ 注意: launch_passive 的窗口按键会同时触发 MuJoCo 自带快捷键
          (数字键切 geomgroup 可见性、方向键步进/调速), 无法在 Python 侧屏蔽。
          推荐改用**终端键盘** (run() 里的 stdin 线程), 与真机 main.cpp 一致,
          按键不进 MuJoCo 窗口, 无冲突。此回调保留为窗口内备用。
        """
        # 方向键映射到 w/s/q/e 等价指令 (上下=vx, 左右=wz)
        if keycode == 265:
            return self._cmd_key("w")
        if keycode == 264:
            return self._cmd_key("s")
        if keycode == 263:
            return self._cmd_key("q")
        if keycode == 262:
            return self._cmd_key("e")
        if keycode >= 256:
            return
        self._cmd_key(chr(keycode).lower())

    def _active_head_limits(self):
        """Return head-command limits for the policy that is currently running."""
        return self.walk_max_head if self.state == "RL_WALK" else self.stand_max_head

    def _active_policy_head_command(self):
        """Bound persistent commands before they enter the active policy observation."""
        limits = self._active_head_limits()
        command = self.head_cmd
        if self.state == "RL_STAND" and self._stand_to_walk_stage is not None:
            command = self._stand_to_walk_effective_head_cmd
        return np.clip(command, -limits, limits)

    def _cmd_key(self, c):
        """处理单个按键字符 (已转小写)。窗口回调与终端线程共用。

        与真机 src/main.cpp 一致: 0/1/2/3=状态 9=阻尼 r=重置 p=电机开关
        5/6=viewer 配置推力正向/反向（仅仿真）
        w/s=vx± a/d=vy± q/e=wz± x=速度清零
        """
        head_limits = self._active_head_limits()
        if c == "0":
            self.start_real_sit_align()
        elif c == "p":
            if not self.bridge:
                print("[real] 未连接真机桥接, 忽略电机输出开关")
                return
            self.real_output_enabled = not self.real_output_enabled
            if self.real_panel:
                self.real_panel.set_enabled(self.real_output_enabled)
            if self.real_output_enabled:
                self.real_cmd_q = self.bridge.get_state()[0][:self.nj].copy()
            print(f"[real] motor output {'ENABLED' if self.real_output_enabled else 'DISABLED'}")
        elif c == "1":
            # SIT: 起立脚本；RL 中: 切到站立模型 (RL_STAND)
            if self.state == "SIT":
                if self.real_sit_active:
                    print("[real] 真机仍在缓慢回蹲姿, 完成后再按 1")
                    return
                if self.bridge and not self.real_sit_done:
                    print("[real] 先按 0 让真机缓慢回蹲姿; 到位后再按 1")
                    return
                self.stand_start = self.data.qpos[self.q_adr].copy()
                self.state, self.state_time = "STAND_UP", 0.0
                print("[FSM] SIT -> STAND_UP (起立后自动进入 RL_STAND 站立模型)")
            elif self.state in ("RL_STAND", "RL_WALK"):
                self._switch_rl_state("RL_STAND")
        elif c == "2":
            # 切到行走模型 (RL_WALK)
            if self.state in ("RL_STAND", "RL_WALK"):
                self._switch_rl_state("RL_WALK")
        elif c == "3":
            # 兼容旧键位: 回站立模型 (= 按 1)
            if self.state in ("RL_STAND", "RL_WALK"):
                self._switch_rl_state("RL_STAND")
        elif c == "5":
            self._request_viewer_push(+1.0)
        elif c == "6":
            self._request_viewer_push(-1.0)
        elif c == "9":
            self.state = "DAMPING"
            self._rl_switch_requests = queue.SimpleQueue()
            self._clear_walk_stop_transition()
            self._clear_stand_to_walk_transition()
            self._effective_walk_cmd[:] = 0.0
            self._rl_gain_blend_active = False
            self._rl_gain_blend_elapsed = 0.0
            self._rl_gain_blend_ratio = 1.0
            self.disable_real_output("DAMPING")
            print("[FSM] -> DAMPING")
        elif c == "r":
            self.reset()
            self.disable_real_output("RESET")
        elif c == "x":
            self.cmd[:] = 0
        elif c == "w":
            self.cmd[0] = min(self.cmd[0] + 0.1, self.max_vel[0])
        elif c == "s":
            self.cmd[0] = max(self.cmd[0] - 0.1, -self.max_vel[0])
        elif c == "a":
            self.cmd[1] = min(self.cmd[1] + 0.1, self.max_vel[1])
        elif c == "d":
            self.cmd[1] = max(self.cmd[1] - 0.1, -self.max_vel[1])
        elif c == "q":
            self.cmd[2] = min(self.cmd[2] + 0.1, self.max_vel[2])
        elif c == "e":
            self.cmd[2] = max(self.cmd[2] - 0.1, -self.max_vel[2])
        # ---- 头部命令（RL_BALANCE / RL_WALK 均生效，脖子随时可动）----
        # i/k=点头pitch± j/l=摇头yaw± u/o=歪头roll± n/m=头高± h=头命令清零
        elif c in "ikjluonmh" and self._stand_to_walk_stage is not None:
            print("[FSM] stand->walk 回正中，暂不接受新的 head 命令")
            return
        elif c == "i":
            self.head_cmd[1] = min(self.head_cmd[1] + 0.1, head_limits[1])
        elif c == "k":
            self.head_cmd[1] = max(self.head_cmd[1] - 0.1, -head_limits[1])
        elif c == "j":
            self.head_cmd[2] = min(self.head_cmd[2] + 0.1, head_limits[2])
        elif c == "l":
            self.head_cmd[2] = max(self.head_cmd[2] - 0.1, -head_limits[2])
        elif c == "u":
            self.head_cmd[3] = min(self.head_cmd[3] + 0.1, head_limits[3])
        elif c == "o":
            self.head_cmd[3] = max(self.head_cmd[3] - 0.1, -head_limits[3])
        elif c == "n":
            self.head_cmd[0] = min(self.head_cmd[0] + 0.005, head_limits[0])
        elif c == "m":
            self.head_cmd[0] = max(self.head_cmd[0] - 0.005, -head_limits[0])
        elif c == "h":
            self.head_cmd[:] = 0.0
        # ---- 站立躯干命令（仅 RL_STAND 生效，微调站姿）----
        # t/g=前后倾pitch± v/c=偏航yaw± y/b=侧倾roll± f/z=高度h±
        elif c in "tgvcybfz" and self._stand_to_walk_stage is not None:
            print("[FSM] stand->walk 回正中，暂不接受新的 torso 命令")
            return
        elif c == "t":
            self.torso_cmd[1] = min(self.torso_cmd[1] + 0.05, self.torso_command_max[1])
        elif c == "g":
            self.torso_cmd[1] = max(self.torso_cmd[1] - 0.05, self.torso_command_min[1])
        elif c == "v":
            self.torso_cmd[2] = min(self.torso_cmd[2] + 0.05, self.torso_command_max[2])
        elif c == "c":
            self.torso_cmd[2] = max(self.torso_cmd[2] - 0.05, self.torso_command_min[2])
        elif c == "y":
            self.torso_cmd[3] = min(self.torso_cmd[3] + 0.05, self.torso_command_max[3])
        elif c == "b":
            self.torso_cmd[3] = max(self.torso_cmd[3] - 0.05, self.torso_command_min[3])
        elif c == "f":
            self.torso_cmd[0] = min(self.torso_cmd[0] + 0.005, self.torso_command_max[0])
        elif c == "z":
            self.torso_cmd[0] = max(self.torso_cmd[0] - 0.005, self.torso_command_min[0])
        else:
            return
        if c in "wsadqex":
            tag = "" if self.state == "RL_WALK" else "  (仅 RL_WALK 生效, 先按2)"
            print(
                f"cmd_target = vx={self.cmd[0]:+.2f} vy={self.cmd[1]:+.2f} "
                f"wz={self.cmd[2]:+.2f}{tag}"
            )
        if c in "ikjluonmh":
            print(f"head_cmd = Δh={self.head_cmd[0]:+.3f} pitch={self.head_cmd[1]:+.2f} "
                  f"yaw={self.head_cmd[2]:+.2f} roll={self.head_cmd[3]:+.2f}")
        if c in "tgvcybfz":
            tag = "" if self.state == "RL_STAND" else "  (仅 RL_STAND 生效, 先按1)"
            print(f"torso_cmd = h={self.torso_cmd[0]:+.3f} pitch={self.torso_cmd[1]:+.2f} "
                  f"yaw={self.torso_cmd[2]:+.2f} roll={self.torso_cmd[3]:+.2f}{tag}")

    def _apply_puppeteering_snapshot(self, snapshot: GamepadSnapshot):
        """Apply one consistent controller snapshot on the 200 Hz control thread."""
        was_connected = self._puppeteer_connected
        mapped = self.puppeteer.update(
            snapshot,
            self.control_dt,
            active_walking=self.state == "RL_WALK",
        )
        self._puppeteer_walk_requested = mapped.walk_requested
        self._puppeteer_connected = mapped.connected
        self._puppeteer_full_speed = mapped.full_speed
        self._puppeteer_explicit_stand_request = mapped.stand_requested
        if (
            mapped.stand_requested
            or mapped.start_requested
            or (was_connected and not mapped.connected)
        ) and self._puppeteer_blocked_target == "RL_STAND":
            self._puppeteer_blocked_target = None
        if mapped.start_requested and self.state == "SIT":
            self._cmd_key("1")
        if self.state == "RL_STAND" and self._stand_to_walk_stage is not None:
            self.head_cmd[:] = 0.0
        elif self.state == "RL_STAND" and mapped.walk_requested:
            # R1 changes the left stick from standing posture to walking velocity. Preserve
            # the head pose that was actually active before this remapping frame; otherwise
            # the standing counter-gaze term creates a spurious head recenter request.
            pass
        else:
            self.head_cmd[:] = mapped.head_command
        if mapped.walk_requested:
            self.cmd[:] = mapped.walk_command
            self.torso_cmd[:] = 0.0
        else:
            self.cmd[:] = 0.0
            if self.state == "RL_STAND" and self._stand_to_walk_stage is None:
                self.torso_cmd[:] = mapped.torso_command
            else:
                self.torso_cmd[:] = 0.0

    def _poll_gamepad(self):
        if not self.gamepad_enabled:
            return
        snapshot = self.gamepad.snapshot() if self.gamepad is not None else GamepadSnapshot.zero()
        self._apply_puppeteering_snapshot(snapshot)

    def _update_puppeteering_switch(self, q, dq, quat, gyro):
        """Turn the persistent R1 mode request into safe policy-switch requests."""
        if not self.gamepad_enabled:
            return

        if (
            self.state == "RL_STAND"
            and self._stand_to_walk_stage is None
            and not self._rl_gain_blend_active
        ):
            stable, _ = self._stand_to_walk_stable(q, dq, quat, gyro)
            if stable:
                self._stand_ready_elapsed += self.control_dt
            else:
                self._stand_ready_elapsed = 0.0
        else:
            self._stand_ready_elapsed = 0.0

        desired = "RL_WALK" if self._puppeteer_walk_requested else "RL_STAND"
        if self._puppeteer_blocked_target is not None and self._puppeteer_blocked_target != desired:
            self._puppeteer_blocked_target = None
        if self._puppeteer_blocked_target == desired:
            return
        if self.state not in ("RL_STAND", "RL_WALK"):
            return

        if desired == "RL_WALK" and self.walk_policy is None:
            self._puppeteer_blocked_target = desired
            print("[gamepad] walking requested but no walking policy is loaded")
            return
        if desired == "RL_STAND" and self.stand_policy is None:
            self._puppeteer_blocked_target = desired
            print("[gamepad] standing requested but no standing policy is loaded")
            return

        if desired == "RL_WALK":
            if self.state == "RL_STAND" and self._stand_to_walk_stage is None:
                preconfirmed = (
                    self._stand_ready_elapsed >= self.stand_to_walk_stable_confirm_duration
                )
                self._switch_rl_state(
                    "RL_WALK",
                    source="gamepad",
                    preserve_walk_command=True,
                    preconfirmed_stable=preconfirmed,
                )
            elif (
                self.state == "RL_WALK"
                and self.pending_rl_state == "RL_STAND"
                and self._walk_stop_source == "gamepad"
            ):
                self._switch_rl_state(
                    "RL_WALK", source="gamepad", preserve_walk_command=True
                )
        else:
            if self.state == "RL_WALK" and self.pending_rl_state != "RL_STAND":
                self._switch_rl_state("RL_STAND", source="gamepad")
            elif (
                self.state == "RL_STAND"
                and self._stand_to_walk_stage is not None
                and not self._stand_to_walk_cancelled
                and self._stand_to_walk_source == "gamepad"
            ):
                self._switch_rl_state("RL_STAND", source="gamepad")

    def _switch_rl_state(
        self,
        target,
        *,
        source="operator",
        preserve_walk_command=False,
        preconfirmed_stable=False,
    ):
        """Queue a policy switch for the next 200 Hz control-loop boundary.

        Keyboard input may arrive on a background thread. It must not mutate policy history or
        FOH/LPF buffers while inference is running. The main loop consumes this scalar request.
        Both directions are completed by real-signal-compatible decel/recenter/settle sequences.
        """
        request = (
            target,
            str(source),
            bool(preserve_walk_command),
            bool(preconfirmed_stable),
        )
        self._rl_switch_requests.put(request)
        print(f"[FSM] {source} switch request queued: {self.state} -> {target}")

    def _clear_walk_stop_transition(self):
        self.pending_rl_state = None
        self._walk_stop_stage = None
        self._walk_stop_total_elapsed = 0.0
        self._walk_stop_stage_elapsed = 0.0
        self._walk_stop_stable_elapsed = 0.0
        self._walk_stop_block_reason = "idle"
        self._walk_stop_source = None

    def _clear_stand_to_walk_transition(self):
        self._stand_to_walk_stage = None
        self._stand_to_walk_total_elapsed = 0.0
        self._stand_to_walk_stage_elapsed = 0.0
        self._stand_to_walk_stable_elapsed = 0.0
        self._stand_to_walk_recenter_duration = 0.0
        self._stand_to_walk_start_torso_cmd[:] = 0.0
        self._stand_to_walk_start_head_cmd[:] = 0.0
        self._stand_to_walk_effective_head_cmd[:] = 0.0
        self._stand_to_walk_block_reason = "idle"
        self._stand_to_walk_cancelled = False
        self._stand_to_walk_source = None
        self._stand_to_walk_preserve_walk_command = False
        self._stand_to_walk_preconfirmed_stable = False

    def _start_gait_from_safe_phase(self, command):
        if self.walk_policy is None:
            return
        # Positive yaw begins a left step, negative yaw a right step; translation alternates.
        if abs(float(command[2])) > 1.0e-4:
            start_left = command[2] > 0.0
        else:
            start_left = bool(np.random.random() < 0.5)
        self.walk_policy.gait_phase = 0.1 if start_left else 0.6

    def _update_walk_command_smoothing(self):
        """Rate-limit policy commands and only cross zero at a safe gait phase."""
        target = np.clip(np.asarray(self.cmd, dtype=float).copy(), -self.max_vel, self.max_vel)
        current = self._effective_walk_cmd.copy()
        was_moving = float(np.max(np.abs(current))) > self.move_command_threshold

        # Commands inside the policy's motion deadband are treated as an explicit stop.
        if float(np.max(np.abs(target))) <= self.move_command_threshold:
            target[:] = 0.0

        reversing = current * target < 0.0
        reducing_magnitude = np.abs(target) < np.abs(current)
        approaching_zero = bool(np.any(reversing | reducing_magnitude))
        rate_limits = np.where(
            reversing | reducing_magnitude,
            self.walk_command_decel_limits,
            self.walk_command_accel_limits,
        )
        max_delta = rate_limits * self.control_dt
        candidate = current + np.clip(target - current, -max_delta, max_delta)

        # Do not let the gait clock freeze in single support. Hold a command just above the
        # motion threshold until a reference double-support window, then continue smoothly
        # through zero (or into the opposite direction) while the safe phase is frozen.
        candidate_peak = float(np.max(np.abs(candidate)))
        current_peak = float(np.max(np.abs(current)))
        if (
            not self._walk_command_safe_zero_active
            and was_moving
            and approaching_zero
            and candidate_peak < self.switch_min_moving_command
        ):
            if self._phase_in_switch_window():
                self._walk_command_safe_zero_active = True
            elif current_peak > 1.0e-9:
                candidate = current / current_peak * self.switch_min_moving_command

        self._effective_walk_cmd[:] = candidate
        is_moving = float(np.max(np.abs(candidate))) > self.move_command_threshold
        if is_moving and not was_moving:
            self._start_gait_from_safe_phase(target)
        if self._walk_command_safe_zero_active:
            target_stopped = float(np.max(np.abs(target))) <= self.move_command_threshold
            reached_zero = float(np.max(np.abs(candidate))) <= 1.0e-9
            moving_with_target = is_moving and float(np.dot(candidate, target)) > 0.0
            if (target_stopped and reached_zero) or moving_with_target:
                self._walk_command_safe_zero_active = False

    def _begin_walk_to_stand(self, source="operator"):
        self._clear_stand_to_walk_transition()
        self.pending_rl_state = "RL_STAND"
        self._walk_stop_source = source
        self._walk_command_safe_zero_active = False
        self._walk_stop_start_cmd[:] = self._effective_walk_cmd
        self._walk_stop_total_elapsed = 0.0
        self._walk_stop_stage_elapsed = 0.0
        self._walk_stop_stable_elapsed = 0.0
        self._walk_stop_block_reason = "decelerating"
        if np.max(np.abs(self._walk_stop_start_cmd)) > self.move_command_threshold:
            self._walk_stop_stage = "DECEL"
            print(
                "[FSM] RL_WALK -> WALK_DECEL "
                f"cmd={np.round(self._walk_stop_start_cmd, 3).tolist()}"
            )
        else:
            self._walk_stop_stage = "ZERO_HOLD"
            self._effective_walk_cmd[:] = 0.0
            print("[FSM] RL_WALK -> WALK_ZERO_HOLD (already zero-speed command)")

    def _cancel_walk_to_stand(self, reason, resume_command):
        stage = self._walk_stop_stage
        source = self._walk_stop_source
        self._clear_walk_stop_transition()
        if not resume_command:
            self.cmd[:] = 0.0
        if source == "gamepad" and not resume_command:
            self._puppeteer_blocked_target = "RL_STAND"
        print(f"[FSM] walk->stand cancelled at {stage}: {reason}")

    def _start_stand_recenter(self):
        self._stand_to_walk_start_torso_cmd[:] = self._effective_torso_cmd
        torso_extent = np.maximum(
            np.abs(self.torso_command_min), np.abs(self.torso_command_max)
        )
        torso_offset = float(
            np.max(
                np.abs(self._stand_to_walk_start_torso_cmd)
                / np.maximum(torso_extent, 1.0e-6)
            )
        )
        head_offset = float(
            np.max(
                np.abs(self._stand_to_walk_start_head_cmd)
                / np.maximum(self.stand_max_head, 1.0e-6)
            )
        )
        normalized_offset = max(torso_offset, head_offset)
        self._stand_to_walk_recenter_duration = (
            self.stand_to_walk_recenter_max_duration * min(1.0, normalized_offset)
        )
        self._stand_to_walk_stage_elapsed = 0.0
        self._stand_to_walk_stable_elapsed = 0.0
        if self._stand_to_walk_recenter_duration <= self.control_dt:
            self._effective_torso_cmd[:] = 0.0
            self._stand_to_walk_effective_head_cmd[:] = 0.0
            self._stand_to_walk_stage = "ZERO_HOLD"
            self._stand_to_walk_block_reason = "minimum neutral-command hold"
            print("[FSM] RL_STAND -> STAND_NEUTRAL_HOLD (already neutral)")
        else:
            self._stand_to_walk_stage = "RECENTER"
            self._stand_to_walk_block_reason = "recentering torso/head command"
            print(
                "[FSM] RL_STAND -> STAND_RECENTER "
                f"duration={self._stand_to_walk_recenter_duration:.2f}s "
                f"torso={np.round(self._stand_to_walk_start_torso_cmd, 3).tolist()} "
                f"head={np.round(self._stand_to_walk_start_head_cmd, 3).tolist()}"
            )

    def _begin_stand_to_walk(
        self,
        source="operator",
        preserve_walk_command=False,
        preconfirmed_stable=False,
    ):
        self._clear_walk_stop_transition()
        self.pending_rl_state = "RL_WALK"
        self._stand_to_walk_source = source
        self._stand_to_walk_preserve_walk_command = bool(preserve_walk_command)
        self._stand_to_walk_preconfirmed_stable = bool(preconfirmed_stable)
        if not preserve_walk_command:
            self.cmd[:] = 0.0
        self._effective_walk_cmd[:] = 0.0
        # Clear user targets immediately, while keeping the commands applied to the standing
        # policy continuous. _start_stand_recenter() ramps both effective commands to zero.
        self._stand_to_walk_start_head_cmd[:] = np.clip(
            self.head_cmd, -self.stand_max_head, self.stand_max_head
        )
        self._stand_to_walk_effective_head_cmd[:] = self._stand_to_walk_start_head_cmd
        self.head_cmd[:] = 0.0
        self.torso_cmd[:] = 0.0
        self._stand_to_walk_total_elapsed = 0.0
        self._stand_to_walk_stage_elapsed = 0.0
        self._stand_to_walk_stable_elapsed = 0.0
        self._stand_to_walk_cancelled = False
        if self._rl_gain_blend_active:
            self._stand_to_walk_stage = "WAIT_GAIN_BLEND"
            self._stand_to_walk_block_reason = "waiting for stand-up policy hand-off"
            print("[FSM] RL_STAND -> STAND_WAIT_GAIN_BLEND")
        else:
            self._start_stand_recenter()
            if self._stand_to_walk_preconfirmed_stable and self._stand_to_walk_stage == "ZERO_HOLD":
                self._stand_to_walk_stage_elapsed = self.stand_to_walk_zero_hold_duration
                self._stand_to_walk_stable_elapsed = self.stand_to_walk_stable_confirm_duration
                self._stand_to_walk_block_reason = "preconfirmed stable standing"

    def _cancel_stand_to_walk(self, reason):
        stage = self._stand_to_walk_stage
        source = self._stand_to_walk_source
        self.pending_rl_state = None
        self.cmd[:] = 0.0
        self._effective_walk_cmd[:] = 0.0
        self.head_cmd[:] = 0.0
        self.torso_cmd[:] = 0.0
        self._stand_to_walk_cancelled = True
        if stage == "ZERO_HOLD":
            self._effective_torso_cmd[:] = 0.0
            self._stand_to_walk_effective_head_cmd[:] = 0.0
            self._clear_stand_to_walk_transition()
        if source == "gamepad":
            self._puppeteer_blocked_target = None
        print(
            f"[FSM] stand->walk cancelled at {stage}: {reason}; "
            "remain RL_STAND and continue to neutral"
        )

    def _phase_in_switch_window(self):
        if self.walk_policy is None:
            return True
        phase = float(self.walk_policy.gait_phase % 1.0)
        for lo, hi in self.switch_double_support_phase_windows:
            if lo <= hi:
                inside = lo <= phase <= hi
            else:
                inside = phase >= lo or phase <= hi
            if inside:
                return True
        return False

    def _walk_to_stand_stable(self, q, dq, quat, gyro):
        """Return a fail-closed stability decision using real-robot-available signals only."""
        arrays = (q, dq, quat, gyro, self.rl_target)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            return False, "non-finite sensor/target"

        projected_gravity = quat_rotate_inverse_gravity(quat)
        proj_xy = float(np.linalg.norm(projected_gravity[:2]))
        upright = float(-projected_gravity[2])
        gyro_xy = float(np.max(np.abs(gyro[:2])))
        gyro_z = abs(float(gyro[2]))
        dq_rms = float(np.sqrt(np.mean(np.square(dq))))
        dq_max = float(np.max(np.abs(dq)))
        q_error = float(np.max(np.abs(q - self.stand_pose)))
        target_error = float(np.max(np.abs(self.rl_target - q)))

        blocked = []
        if proj_xy >= self.switch_proj_g_xy_max:
            blocked.append(f"proj_xy={proj_xy:.3f}")
        if upright <= self.switch_upright_projection_min:
            blocked.append(f"upright={upright:.3f}")
        if gyro_xy >= self.switch_gyro_xy_max:
            blocked.append(f"gyro_xy={gyro_xy:.3f}")
        if gyro_z >= self.switch_gyro_z_max:
            blocked.append(f"gyro_z={gyro_z:.3f}")
        if dq_rms >= self.switch_joint_vel_rms_max:
            blocked.append(f"dq_rms={dq_rms:.3f}")
        if dq_max >= self.switch_joint_vel_max:
            blocked.append(f"dq_max={dq_max:.3f}")
        if q_error >= self.switch_joint_pos_error_max:
            blocked.append(f"q_err={q_error:.3f}")
        if target_error >= self.switch_target_error_max:
            blocked.append(f"target_err={target_error:.3f}")
        return not blocked, ", ".join(blocked) if blocked else "stable"

    def _stand_to_walk_stable(self, q, dq, quat, gyro):
        stable, reason = self._walk_to_stand_stable(q, dq, quat, gyro)
        if not np.all(np.isfinite(self._effective_torso_cmd)):
            return False, "non-finite torso command"
        torso_error = float(np.max(np.abs(self._effective_torso_cmd)))
        if torso_error > 1.0e-6:
            extra = f"torso_cmd={torso_error:.4f}"
            return False, f"{reason}, {extra}" if reason != "stable" else extra
        return stable, reason

    def _update_stand_to_walk(self, q, dq, quat, gyro, policy_boundary):
        self._stand_to_walk_total_elapsed += self.control_dt
        if self._stand_to_walk_total_elapsed >= self.stand_to_walk_total_timeout:
            reason = self._stand_to_walk_block_reason or "transition timeout"
            source = self._stand_to_walk_source
            preserve_walk_command = self._stand_to_walk_preserve_walk_command
            self.pending_rl_state = None
            if not preserve_walk_command:
                self.cmd[:] = 0.0
            self._effective_walk_cmd[:] = 0.0
            self.head_cmd[:] = 0.0
            self.torso_cmd[:] = 0.0
            if source == "gamepad":
                self._puppeteer_blocked_target = "RL_WALK"
            if self._stand_to_walk_stage in ("WAIT_GAIN_BLEND", "RECENTER"):
                # Do not turn a timeout into the same torso-command jump this FSM prevents.
                # Finish the already-running recenter, then remain on the standing policy.
                self._stand_to_walk_cancelled = True
                self._stand_to_walk_total_elapsed = 0.0
            else:
                self._effective_torso_cmd[:] = 0.0
                self._stand_to_walk_effective_head_cmd[:] = 0.0
                self._clear_stand_to_walk_transition()
            print(
                "[FSM] stand->walk timeout after "
                f"{self.stand_to_walk_total_timeout:.2f}s ({reason}); remain RL_STAND neutral"
            )
            return

        if self._stand_to_walk_stage == "WAIT_GAIN_BLEND":
            if not self._rl_gain_blend_active:
                self._start_stand_recenter()
            return

        if self._stand_to_walk_stage == "RECENTER":
            self._stand_to_walk_stage_elapsed += self.control_dt
            progress = min(
                1.0,
                self._stand_to_walk_stage_elapsed
                / self._stand_to_walk_recenter_duration,
            )
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))
            self._effective_torso_cmd[:] = (
                self._stand_to_walk_start_torso_cmd * (1.0 - blend)
            )
            self._stand_to_walk_effective_head_cmd[:] = (
                self._stand_to_walk_start_head_cmd * (1.0 - blend)
            )
            if progress >= 1.0:
                self._effective_torso_cmd[:] = 0.0
                self._stand_to_walk_effective_head_cmd[:] = 0.0
                if self._stand_to_walk_cancelled:
                    self._clear_stand_to_walk_transition()
                    print("[FSM] stand->walk cancellation reached neutral; remain RL_STAND")
                else:
                    self._stand_to_walk_stage = "ZERO_HOLD"
                    self._stand_to_walk_stage_elapsed = 0.0
                    self._stand_to_walk_stable_elapsed = 0.0
                    self._stand_to_walk_block_reason = "minimum neutral-command hold"
                    print("[FSM] STAND_RECENTER -> STAND_NEUTRAL_HOLD")
            return

        if self._stand_to_walk_stage != "ZERO_HOLD":
            return

        self._effective_torso_cmd[:] = 0.0
        self._stand_to_walk_effective_head_cmd[:] = 0.0
        if self._stand_to_walk_cancelled:
            self._clear_stand_to_walk_transition()
            return
        self._stand_to_walk_stage_elapsed += self.control_dt
        if self._stand_to_walk_stage_elapsed < self.stand_to_walk_zero_hold_duration:
            self._stand_to_walk_stable_elapsed = 0.0
            return

        stable, reason = self._stand_to_walk_stable(q, dq, quat, gyro)
        self._stand_to_walk_block_reason = reason
        if stable:
            self._stand_to_walk_stable_elapsed += self.control_dt
        else:
            self._stand_to_walk_stable_elapsed = 0.0
        if (
            self._stand_to_walk_stable_elapsed
            >= self.stand_to_walk_stable_confirm_duration
            and policy_boundary
        ):
            print(
                "[FSM] STAND_NEUTRAL_HOLD stable "
                f"for {self._stand_to_walk_stable_elapsed:.2f}s -> RL_WALK"
            )
            self._complete_rl_switch("RL_WALK")

    def _update_walk_to_stand(self, q, dq, quat, gyro, policy_boundary):
        self._walk_stop_total_elapsed += self.control_dt
        if self._walk_stop_total_elapsed >= self.switch_total_timeout:
            reason = self._walk_stop_block_reason or "transition timeout"
            self._cancel_walk_to_stand(
                f"timeout after {self._walk_stop_total_elapsed:.2f}s ({reason})",
                resume_command=False,
            )
            return

        if self._walk_stop_stage == "DECEL":
            self._walk_stop_stage_elapsed += self.control_dt
            progress = min(1.0, self._walk_stop_stage_elapsed / self.switch_decel_duration)
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))
            candidate = self._walk_stop_start_cmd * (1.0 - blend)
            candidate_peak = float(np.max(np.abs(candidate)))
            start_peak = float(np.max(np.abs(self._walk_stop_start_cmd)))
            if candidate_peak < self.switch_min_moving_command and start_peak > 1.0e-9:
                candidate = (
                    self._walk_stop_start_cmd / start_peak * self.switch_min_moving_command
                )
            self._effective_walk_cmd[:] = candidate
            if progress >= 1.0:
                self._walk_stop_stage = "WAIT_PHASE"
                self._walk_stop_stage_elapsed = 0.0
                self._walk_stop_block_reason = "waiting for expected double support"
                print("[FSM] WALK_DECEL -> WALK_WAIT_DOUBLE_SUPPORT")
            return

        if self._walk_stop_stage == "WAIT_PHASE":
            if self._phase_in_switch_window():
                self._walk_stop_start_cmd[:] = self._effective_walk_cmd
                self._walk_stop_stage = "FINAL_DECEL"
                self._walk_stop_stage_elapsed = 0.0
                self._walk_stop_stable_elapsed = 0.0
                self._walk_stop_block_reason = "final double-support deceleration"
                phase = self.walk_policy.gait_phase if self.walk_policy is not None else 0.0
                print(
                    "[FSM] WALK_WAIT_DOUBLE_SUPPORT -> WALK_FINAL_DECEL "
                    f"phase={phase:.3f}"
                )
            return

        if self._walk_stop_stage == "FINAL_DECEL":
            self._walk_stop_stage_elapsed += self.control_dt
            progress = min(
                1.0,
                self._walk_stop_stage_elapsed / self.switch_final_decel_duration,
            )
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))
            self._effective_walk_cmd[:] = self._walk_stop_start_cmd * (1.0 - blend)
            if progress >= 1.0:
                self._effective_walk_cmd[:] = 0.0
                self._walk_stop_stage = "ZERO_HOLD"
                self._walk_stop_stage_elapsed = 0.0
                self._walk_stop_block_reason = "minimum zero-command hold"
                print("[FSM] WALK_FINAL_DECEL -> WALK_ZERO_HOLD")
            return

        if self._walk_stop_stage != "ZERO_HOLD":
            return

        self._effective_walk_cmd[:] = 0.0
        self._walk_stop_stage_elapsed += self.control_dt
        if self._walk_stop_stage_elapsed < self.switch_zero_hold_duration:
            self._walk_stop_stable_elapsed = 0.0
            return

        stable, reason = self._walk_to_stand_stable(q, dq, quat, gyro)
        self._walk_stop_block_reason = reason
        if stable:
            self._walk_stop_stable_elapsed += self.control_dt
        else:
            self._walk_stop_stable_elapsed = 0.0
        if (
            self._walk_stop_stable_elapsed >= self.switch_stable_confirm_duration
            and policy_boundary
        ):
            print(
                "[FSM] WALK_ZERO_HOLD stable "
                f"for {self._walk_stop_stable_elapsed:.2f}s -> RL_STAND"
            )
            self._complete_rl_switch("RL_STAND")

    def _complete_rl_switch(self, target):
        """Complete a policy switch while preserving action and setpoint continuity."""
        source_policy = self.policy
        switch_source = (
            self._stand_to_walk_source if target == "RL_WALK" else self._walk_stop_source
        )
        preserve_walk_command = (
            target == "RL_WALK" and self._stand_to_walk_preserve_walk_command
        )
        requested_walk_command = np.clip(
            np.asarray(self.cmd, dtype=float).copy(), -self.max_vel, self.max_vel
        )
        if target == "RL_WALK":
            target_policy = self.walk_policy
            if preserve_walk_command:
                self.cmd[:] = requested_walk_command
            else:
                self.cmd[:] = 0.0
            self._effective_walk_cmd[:] = 0.0
            self.head_cmd[:] = 0.0
            self.torso_cmd[:] = 0.0
            self._effective_torso_cmd[:] = 0.0
            label = "walk model"
        else:  # RL_STAND
            target_policy = self.stand_policy or self.walk_policy
            # Model switches do not restore a previous torso pose. Enter standing at the
            # neutral torso command and let the standing policy hold it directly.
            self.cmd[:] = 0.0
            self._effective_walk_cmd[:] = 0.0
            self.torso_cmd[:] = 0.0
            self._effective_torso_cmd[:] = 0.0
            label = "stand model" if self.stand_policy else "stand(回退行走模型)"

        if source_policy is not None:
            previous_actions = source_policy.last_actions.copy()
            previous_previous_actions = source_policy.last_last_actions.copy()
        else:
            previous_actions = None
            previous_previous_actions = None
        if target_policy is not None:
            target_policy.reset(previous_actions, previous_previous_actions)

        # Keep the target actually applied by the outgoing policy. Starting from measured q
        # would introduce a setpoint and torque discontinuity even though q itself is smooth.
        current_leg_target = self.rl_target.copy()
        current_neck_target = self.neck_target.copy()
        self.policy_target_prev = current_leg_target.copy()
        self.policy_target_next = current_leg_target.copy()
        self.filtered_policy_target = current_leg_target.copy()
        self.neck_policy_target_prev = current_neck_target.copy()
        self.neck_policy_target_next = current_neck_target.copy()
        self.filtered_neck_target = current_neck_target.copy()
        self.policy_substep = self.decimation
        self.rl_tick = 0
        self.policy = target_policy
        self.state = target
        self._effective_walk_cmd[:] = 0.0
        self._walk_command_safe_zero_active = False
        self._stand_ready_elapsed = 0.0
        self._clear_walk_stop_transition()
        self._clear_stand_to_walk_transition()
        source_text = f", source={switch_source}" if switch_source else ""
        print(f"[FSM] -> {target} ({label}{source_text})")

    def _update_pending_rl_switch(self, q, dq, quat, gyro):
        """Consume thread-safe requests and finish transitions at a control boundary."""
        policy_boundary = self.rl_tick % self.decimation == 0
        request = None
        while True:
            try:
                request = self._rl_switch_requests.get_nowait()
            except queue.Empty:
                break
        if request is not None:
            if isinstance(request, str):
                requested, source, preserve_walk_command, preconfirmed_stable = (
                    request,
                    "operator",
                    False,
                    False,
                )
            else:
                requested, source, preserve_walk_command, preconfirmed_stable = request
            if self.state not in ("RL_STAND", "RL_WALK"):
                self._clear_walk_stop_transition()
                self._clear_stand_to_walk_transition()
                return
            if requested == self.state:
                if self.state == "RL_WALK" and self.pending_rl_state == "RL_STAND":
                    self._cancel_walk_to_stand("operator requested walk", resume_command=True)
                elif self.state == "RL_STAND" and self._stand_to_walk_stage is not None:
                    self._cancel_stand_to_walk("operator requested stand")
                elif self.pending_rl_state is not None:
                    self._clear_walk_stop_transition()
                    print(f"[FSM] pending switch cancelled; remain in {self.state}")
            else:
                if self.state == "RL_WALK" and requested == "RL_STAND":
                    if self.pending_rl_state != "RL_STAND":
                        self._begin_walk_to_stand(source=source)
                elif self.state == "RL_STAND" and requested == "RL_WALK":
                    if self._stand_to_walk_stage is None:
                        self._begin_stand_to_walk(
                            source=source,
                            preserve_walk_command=preserve_walk_command,
                            preconfirmed_stable=preconfirmed_stable,
                        )
                    else:
                        self.pending_rl_state = "RL_WALK"
                        self._stand_to_walk_cancelled = False
                        self._stand_to_walk_source = source
                        self._stand_to_walk_preserve_walk_command = preserve_walk_command
                        self._stand_to_walk_preconfirmed_stable = preconfirmed_stable
                        print("[FSM] stand->walk transition resumed")
                else:
                    self.pending_rl_state = requested

        if self.state == "RL_WALK" and self.pending_rl_state == "RL_STAND":
            self._update_walk_to_stand(q, dq, quat, gyro, policy_boundary)
        elif self.state == "RL_STAND" and self._stand_to_walk_stage is not None:
            self._update_stand_to_walk(q, dq, quat, gyro, policy_boundary)
        elif self.pending_rl_state is not None and policy_boundary:
            self._complete_rl_switch(self.pending_rl_state)
        elif self.state == "RL_WALK":
            self._update_walk_command_smoothing()

    # ---------- 真机输出 ----------
    def disable_real_output(self, reason=""):
        if not self.bridge:
            return
        self.real_output_enabled = False
        self.real_sit_active = False
        self.real_sit_done = False
        self.real_cmd_q = self.bridge.get_state()[0][:self.nj].copy()
        self.bridge.set_enabled(False)
        self.bridge.set_target(self.real_cmd_q, np.zeros(self.nj), np.full(self.nj, self.damping_kd))
        if self.real_panel:
            self.real_panel.set_enabled(False)
        suffix = f" ({reason})" if reason else ""
        print(f"[real] motor output DISABLED{suffix}; press 0 to re-home before standing")

    def start_real_sit_align(self):
        if not self.bridge:
            print("[real] 未连接真机桥接, 忽略蹲姿触发")
            return
        q_real = self.bridge.get_state()[0][:self.nj]
        self.real_cmd_q = q_real.copy()
        self.real_sit_active = True
        self.real_sit_done = False
        self.real_sit_last_print = 0.0
        self.reset()
        if not self.real_output_enabled and self.real_panel:
            self.real_panel.set_enabled(False)
        print("[real] 开始缓慢移动到蹲姿; 需要电机输出使能才会实际动作")

    def update_real_output(self):
        if not self.bridge:
            return

        if self.real_panel:
            self.real_output_enabled = self.real_panel.is_enabled()

        q_real, _, tau_real = self.bridge.get_state()
        q_real = q_real[:self.nj]
        tau_real = tau_real[:self.nj]

        if not self.real_output_enabled:
            self.real_cmd_q = q_real.copy()

        if self.real_sit_active:
            target = np.clip(self.sit_pose, self.joint_lower, self.joint_upper)
            kp, kd = self.real_follow_kp, self.real_follow_kd
            cmd_err = float(np.max(np.abs(target - self.real_cmd_q)))
            real_errs = np.abs(target - q_real)
            real_err = float(np.max(real_errs))
            worst = int(np.argmax(real_errs))
            now = time.time()
            if now - self.real_sit_last_print > 1.0:
                self.real_sit_last_print = now
                en = "ON" if self.real_output_enabled else "OFF"
                print(f"[real] SIT ALIGN enable={en} cmd_err={cmd_err:.3f} "
                      f"real_err={real_err:.3f}/{self.real_sit_ready_tol:.3f} "
                      f"worst={LEG_JOINTS[worst]}")
            if (self.real_output_enabled and
                    cmd_err <= self.real_sit_cmd_tol and
                    real_err <= self.real_sit_ready_tol):
                self.real_sit_active = False
                self.real_sit_done = True
                print("[real] 真机已到蹲姿附近, 可以按 1 起立")
        else:
            target = self.last_target
            if self.state in ("SIT", "STAND_UP", "RL_STAND", "RL_WALK"):
                kp, kd = self.real_follow_kp, self.real_follow_kd
            else:
                kp, kd = self.last_kp, self.last_kd

        slew = float(self.real_cfg.get("slew_rate", 0.5))
        step = slew * self.control_dt
        if self.real_output_enabled:
            self.real_cmd_q += np.clip(target - self.real_cmd_q, -step, step)
        self.real_cmd_q = np.clip(self.real_cmd_q, self.joint_lower, self.joint_upper)

        self.bridge.set_enabled(self.real_output_enabled)
        self.bridge.set_target(self.real_cmd_q, kp, kd)

        if self.real_panel:
            self.real_panel.set_state(self.real_cmd_q, q_real, tau_real,
                                      self.real_sit_active, self.real_sit_done)
            self.real_panel.update()

    # ---------- 控制 ----------
    def control_step(self):
        """以 control_dt 周期调用, 输出关节力矩"""
        self.state_time += self.control_dt
        q, dq, quat, gyro = self.get_obs_raw()

        # 姿态保护
        if self.state in ("STAND_UP", "RL_STAND", "RL_WALK"):
            if quat_rotate_inverse_gravity(quat)[2] > -0.5:
                self.state = "DAMPING"
                self._rl_switch_requests = queue.SimpleQueue()
                self._clear_walk_stop_transition()
                self._clear_stand_to_walk_transition()
                self._effective_walk_cmd[:] = 0.0
                self._rl_gain_blend_active = False
                self._rl_gain_blend_elapsed = 0.0
                self._rl_gain_blend_ratio = 1.0
                print("[FSM] attitude protect -> DAMPING")

        self._poll_gamepad()
        self._update_puppeteering_switch(q, dq, quat, gyro)
        self._update_pending_rl_switch(q, dq, quat, gyro)

        if self.state == "SIT":
            target, kp, kd = self.sit_pose, self.fixed_kp, self.fixed_kd
        elif self.state == "STAND_UP":
            r = min(self.state_time / self.stand_duration, 1.0)
            a = 0.5 * (1 - np.cos(np.pi * r))
            target = self.stand_start * (1 - a) + self.stand_pose * a
            kp, kd = self.fixed_kp, self.fixed_kd
            if r >= 1.0:
                self._remove_stool()  # 站立后移除坐凳, 防止绊脚
                # 起立完成 → 站立模型 (RL_STAND)。无站立模型时回退到行走模型。
                self.policy = self.stand_policy or self.walk_policy
                self.rl_target = self.stand_pose.copy()
                self.policy_target_prev = self.stand_pose.copy()
                self.policy_target_next = self.stand_pose.copy()
                self.filtered_policy_target = self.stand_pose.copy()
                self.neck_target = self.neck_default.copy()
                self.neck_policy_target_prev = self.neck_default.copy()
                self.neck_policy_target_next = self.neck_default.copy()
                self.filtered_neck_target = self.neck_default.copy()
                self.policy_substep = self.decimation
                self.rl_tick = 0
                if self.policy:
                    self.policy.reset()
                # Start in the converged standing frame used by training: feet center/heading.
                _, _, feet_center, feet_heading, _ = self.path_frame_truth()
                self.path_frame.reset(feet_center, feet_heading)
                target_torso_command = self.torso_cmd.copy()
                self._effective_torso_cmd[:] = self._stand_command_from_reliable_pose(
                    target_torso_command
                )
                self._rl_gain_blend_start_torso_command = self._effective_torso_cmd.copy()
                self._rl_gain_blend_target_torso_command = target_torso_command
                self.state, self.state_time = "RL_STAND", 0.0
                self._rl_gain_blend_active = self.rl_gain_blend_duration > 0.0
                self._rl_gain_blend_elapsed = 0.0
                self._rl_gain_blend_ratio = 0.0 if self._rl_gain_blend_active else 1.0
                self._rl_gain_blend_start_target = self.stand_pose.copy()
                self._rl_gain_blend_start_neck_target = self.neck_default.copy()
                if not self._rl_gain_blend_active:
                    self._effective_torso_cmd[:] = self._rl_gain_blend_target_torso_command
                label = "stand model" if self.stand_policy else "stand(回退行走模型)"
                print(f"[FSM] STAND_UP -> RL_STAND ({label})")
        elif self.state in ("RL_STAND", "RL_WALK"):
            if self.state == "RL_STAND" and self._rl_gain_blend_active:
                blend = self._rl_gain_blend_ratio
                self._effective_torso_cmd[:] = (
                    self._rl_gain_blend_start_torso_command * (1.0 - blend)
                    + self._rl_gain_blend_target_torso_command * blend
                )
            elif self.state == "RL_STAND" and self._stand_to_walk_stage is None:
                self._effective_torso_cmd[:] = self.torso_cmd
            # 按状态选活动模型：RL_WALK→行走模型(cmd 速度)，RL_STAND→站立模型(torso 命令)
            if self.state == "RL_WALK":
                self.policy = self.walk_policy
                cmd = self._effective_walk_cmd
                torso_cmd = np.zeros(4, dtype=np.float32)
            else:  # RL_STAND
                self.policy = self.stand_policy or self.walk_policy
                cmd = np.zeros(3)
                torso_cmd = self._effective_torso_cmd
            if self.policy is None:
                target, kp, kd = self.rl_target, self.kp, self.kd  # 无策略兜底: 锁站姿
            else:
                fresh_policy = self.rl_tick % self.decimation == 0
                if fresh_policy:
                    # path frame 每策略步推进一次（用 MuJoCo 真值；真机侧改状态估计器）
                    base_xy, head_yaw, feet_center, feet_heading, lin_vel_b = self.path_frame_truth()
                    # 只有行走且命令超阈值才平移 path frame；站立走收敛分支
                    moving = self.state == "RL_WALK" and (
                        float(np.max(np.abs(cmd[:2]))) > self.move_command_threshold
                        or abs(float(cmd[2])) > self.move_command_threshold
                    )
                    self.path_frame.step(
                        self.policy.policy_dt,
                        cmd,
                        moving,
                        base_xy,
                        head_yaw,
                        feet_center,
                        feet_heading,
                    )
                    pos_pf, yaw_pf = self.path_frame.base_in_path_frame(base_xy, head_yaw)
                    neck_q = self.data.qpos[self.neck_q_adr].copy()
                    neck_dq = self.data.qvel[self.neck_v_adr].copy()
                    # Keyboard commands persist across policy switches. Clamp again at the
                    # inference boundary so the walking policy never receives a standing-only
                    # head command outside its training distribution.
                    policy_head_cmd = self._active_policy_head_command()
                    next_target, next_neck_target = self.policy.step(
                        q, dq, gyro, quat_rotate_inverse_gravity(quat), cmd, pos_pf, yaw_pf, lin_vel_b,
                        neck_q, neck_dq, policy_head_cmd, torso_cmd,
                    )
                    max_setpoint_deviation = self.motor_tau_max / np.maximum(self.kp, 1.0e-6)
                    next_target = np.clip(next_target, q - max_setpoint_deviation, q + max_setpoint_deviation)
                    next_target = np.clip(next_target, self.joint_lower, self.joint_upper)
                    next_neck_target = np.clip(next_neck_target, self.neck_lower, self.neck_upper)
                    self.policy_target_prev = self.policy_target_next.copy()
                    self.policy_target_next = next_target
                    self.neck_policy_target_prev = self.neck_policy_target_next.copy()
                    self.neck_policy_target_next = next_neck_target
                    self.policy_substep = 0
                self.policy_substep = min(self.decimation, self.policy_substep + 1)
                fraction = self.policy_substep / self.decimation
                interpolated_target = self.policy_target_prev + fraction * (
                    self.policy_target_next - self.policy_target_prev
                )
                self.filtered_policy_target += self.target_lowpass_alpha * (
                    interpolated_target - self.filtered_policy_target
                )
                interpolated_neck_target = self.neck_policy_target_prev + fraction * (
                    self.neck_policy_target_next - self.neck_policy_target_prev
                )
                self.filtered_neck_target += self.target_lowpass_alpha * (
                    interpolated_neck_target - self.filtered_neck_target
                )
                self.rl_target = self.filtered_policy_target.copy()
                self.neck_target = self.filtered_neck_target.copy()
                self.rl_tick += 1
                target, kp, kd = self.rl_target, self.kp, self.kd
                if fresh_policy and self.run_logger is not None:
                    tilt = quat_rotate_inverse_gravity(quat)
                    _, vel_b = self.base_linear_velocities()
                    self.run_logger.log(
                        self.data.time, self.state, cmd,
                        float(self.data.qpos[2]), tilt, vel_b, gyro, q, dq,
                        self.policy.last_raw_actions, self.policy.last_actions,
                        self.rl_target,
                    )
            if self._rl_gain_blend_active:
                self._rl_gain_blend_elapsed += self.control_dt
                progress = min(
                    1.0, self._rl_gain_blend_elapsed / self.rl_gain_blend_duration
                )
                blend = 0.5 * (1.0 - np.cos(np.pi * progress))
                self._rl_gain_blend_ratio = blend
                target = self._rl_gain_blend_start_target * (1.0 - blend) + target * blend
                kp = self.fixed_kp * (1.0 - blend) + self.kp * blend
                kd = self.fixed_kd * (1.0 - blend) + self.kd * blend
                if progress >= 1.0:
                    self._rl_gain_blend_active = False
                    self._rl_gain_blend_ratio = 1.0
                    if self.state == "RL_STAND":
                        self._effective_torso_cmd[:] = self._rl_gain_blend_target_torso_command
        else:  # DAMPING
            target, kp, kd = q, np.zeros(self.nj), np.full(self.nj, self.damping_kd)

        self.debug_print_obs(q, dq, quat, gyro, target)

        self.last_target = target.copy()
        self.last_kp = kp.copy()
        self.last_kd = kd.copy()

        tau_m = kp * (target - q) - kd * dq
        if self.state in ("RL_STAND", "RL_WALK"):
            ramp_hi = self.motor_tau_max * (
                self.motor_qd_max - dq
            ) / (self.motor_qd_max - self.motor_qd_tau_max)
            tau_hi = np.where(
                dq <= self.motor_qd_tau_max,
                self.motor_tau_max,
                np.clip(ramp_hi, 0.0, self.motor_tau_max),
            )
            ramp_lo = self.motor_tau_max * (
                self.motor_qd_max + dq
            ) / (self.motor_qd_max - self.motor_qd_tau_max)
            tau_lo = -np.where(
                -dq <= self.motor_qd_tau_max,
                self.motor_tau_max,
                np.clip(ramp_lo, 0.0, self.motor_tau_max),
            )
            friction = self.motor_mu_s * np.tanh(dq / self.motor_qd_s) + self.motor_mu_d * dq
            tau = np.clip(tau_m, tau_lo, tau_hi) - friction
        else:
            tau = np.clip(tau_m, -self.tau_limit, self.tau_limit)

        # 脖子: RL 状态下位置伺服跟随策略输出的脖子目标（阶段1 起脖子参与控制），
        # 其余状态锁默认位。增益对齐训练侧 neck ImplicitActuator（kp50/kd2）。
        neck_q = self.data.qpos[self.neck_q_adr]
        neck_dq = self.data.qvel[self.neck_v_adr]
        if self.state in ("RL_STAND", "RL_WALK"):
            neck_ref = self.neck_target
            blend = self._rl_gain_blend_ratio
            if blend < 1.0:
                neck_ref = (
                    self._rl_gain_blend_start_neck_target * (1.0 - blend)
                    + neck_ref * blend
                )
        else:
            neck_ref = self.neck_default
        neck_ref = np.clip(neck_ref, self.neck_lower, self.neck_upper)
        neck_tau = self.neck_kp * (neck_ref - neck_q) - self.neck_kd * neck_dq
        neck_tau = np.clip(neck_tau, -self.neck_tau_limit, self.neck_tau_limit)

        return tau, neck_tau

    # ---------- 真机联调 / 手动拖动 ----------
    def _run_teleop(self):
        """--real / --manual: sim 作为数字孪生, 滑块拖动关节角。"""
        panel = None
        # 1) 真机桥接
        if self.want_real:
            from real_bridge import RealBridge
            self.bridge = RealBridge(self.cfg)
            if not self.bridge.start():
                print("[real] 启动失败, 退出")
                return
            time.sleep(0.1)  # 等首次读数
            q0 = self.bridge.get_state()[0][:self.nj].copy()
        else:
            q0 = self.stand_pose.copy()
        self.cmd_q = np.asarray(q0, dtype=float).copy()

        # 2) 滑块面板
        if self.want_manual:
            try:
                panel = TeleopPanel(
                    LEG_JOINTS, self.joint_lower, self.joint_upper, q0,
                    on_sync=(lambda: self.bridge.get_state()[0][:self.nj] if self.bridge else None),
                )
            except Exception as e:
                print(f"[teleop] 无法创建滑块面板 ({e}); 请安装 python3-tk")
                if not self.want_real:
                    return

        # 3) 可视化初始姿态 (移除凳子, 固定机身位姿, 只看关节)
        mujoco.mj_resetData(self.model, self.data)
        self.model.geom_pos[self.stool_gid][2] = -1.0
        self.data.qpos[2] = 0.40
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        teleop_kp = np.array(self.real_cfg.get("teleop_kp", [20.0] * self.nj))[:self.nj]
        teleop_kd = np.array(self.real_cfg.get("teleop_kd", [1.0] * self.nj))[:self.nj]
        slew = float(self.real_cfg.get("slew_rate", 0.5))
        loop_dt = 1.0 / float(self.real_cfg.get("poll_hz", 200))

        mode = "REAL+拖动" if (self.want_real and panel) else (
            "REAL镜像(可手动搬动)" if self.want_real else "纯SIM拖动")
        print(f"[teleop] 模式={mode}; 拖动滑块调节关节角, 勾选'使能'后才会下发到真机。")

        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            self._configure_viewer_camera(v)
            while v.is_running() and (panel is None or panel.alive()):
                t0 = time.time()
                targets = panel.get_targets() if panel else self.cmd_q
                targets = np.clip(targets, self.joint_lower, self.joint_upper)
                # 限速: 防止拖动跳变冲击真机
                step = slew * loop_dt
                self.cmd_q += np.clip(targets - self.cmd_q, -step, step)

                if self.want_real:
                    en = panel.is_enabled() if panel else False
                    self.bridge.set_enabled(en)
                    self.bridge.set_target(self.cmd_q, teleop_kp, teleop_kd)
                    q_real, _, tau_real = self.bridge.get_state()
                    q_real = q_real[:self.nj]
                    if panel:
                        panel.set_real(q_real, tau_real[:self.nj])
                    self.data.qpos[self.q_adr] = q_real  # 镜像真机编码器
                else:
                    self.data.qpos[self.q_adr] = self.cmd_q  # 纯 sim 运动学摆姿

                self.data.qpos[self.neck_q_adr] = 0.0
                mujoco.mj_forward(self.model, self.data)
                v.sync()
                if panel:
                    panel.update()
                left = loop_dt - (time.time() - t0)
                if left > 0:
                    time.sleep(left)

        if self.bridge:
            self.bridge.stop()
        if panel:
            panel.close()

    def _start_stdin_thread(self):
        """后台线程: 从终端读单键, 走 _cmd_key, 与真机 main.cpp 一致。

        终端按键不进 MuJoCo 窗口, 因此不会触发 MuJoCo 自带快捷键。
        非 TTY (如管道/重定向) 时跳过, 仅保留窗口回调。
        返回 (thread_or_None, restore_fn)。
        """
        import threading

        if not sys.stdin.isatty():
            print("[sim2sim] stdin 非终端, 跳过终端键盘; 仅窗口内按键可用")
            return None, (lambda: None)

        try:
            import termios
            import tty
        except ImportError:
            return None, (lambda: None)

        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._stdin_alive = True

        def restore():
            termios.tcsetattr(fd, termios.TCSANOW, old_attr)

        def reader():
            import select
            while self._stdin_alive:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._cmd_key(ch.lower())

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        print("[sim2sim] 终端键盘已启用 (聚焦本终端窗口操作, 避免 MuJoCo 快捷键冲突)")
        return t, restore

    def run(self):
        if getattr(self, "probe_policy_only", False):
            return self.probe_policy()
        if self.debug_push_steps:
            return self.run_debug_push()
        if self.want_manual:
            return self._run_teleop()
        sim_steps_per_ctrl = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        print(f"[sim2sim] sim_dt={self.model.opt.timestep} control_dt={self.control_dt} "
              f"({sim_steps_per_ctrl} substeps), policy {'ON' if self.policy else 'OFF'}")
        if np.linalg.norm(self.viewer_push_force) > 1.0e-12:
            print(
                "[viewer_push] configured "
                f"force_w={np.round(self.viewer_push_force, 3).tolist()}N "
                f"duration={self.viewer_push_duration:.3f}s; "
                "press terminal 5 for this direction, 6 for the opposite direction"
            )
        print(__doc__)

        if self.gamepad_enabled:
            device = str(
                self.puppeteering_cfg.get(
                    "device", self.cfg.get("hardware", {}).get("gamepad_device", "/dev/input/js0")
                )
            )
            self.gamepad = LinuxJoystick(device)
            self.gamepad.start()
            if self.puppeteer.walk_button_behavior == "hold":
                print(
                    "[gamepad] deadman mapping enabled: default standing; hold R1 to walk, "
                    "release R1 to request a safe standing transition; START=key 1"
                )
            else:
                print(
                    "[gamepad] paper mapping enabled: default standing; R1 short press toggles "
                    "walk, R1 hold selects full speed, A requests standing, START=key 1"
                )

        if self.want_real:
            from real_bridge import RealBridge
            self.bridge = RealBridge(self.cfg)
            if not self.bridge.start():
                print("[real] 启动失败, 退出")
                return
            time.sleep(0.1)
            self.real_cmd_q = self.bridge.get_state()[0][:self.nj].copy()
            try:
                self.real_panel = RealOutputPanel(LEG_JOINTS, self.start_real_sit_align)
            except Exception as e:
                print(f"[real] 无法创建输出面板 ({e}); 可用键盘 p/0 控制")

        if self.want_runlog and self.policy:
            self.run_logger = RunLogger(self.nj)

        stdin_thread, restore_stdin = self._start_stdin_thread()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_cb) as v:
                self._configure_viewer_camera(v)
                # 记录初始 geom 可见组, 每帧复位; 防止窗口内误按数字键
                # 触发 MuJoCo 自带 geomgroup 切换把机器人 geom 隐藏。
                geomgroup0 = np.array(v.opt.geomgroup).copy()
                while v.is_running() and (self.real_panel is None or self.real_panel.alive()):
                    t0 = time.time()
                    self._consume_viewer_push_requests()
                    tau, neck_tau = self.control_step()
                    self.update_real_output()
                    for _ in range(sim_steps_per_ctrl):
                        self.data.qfrc_applied[:] = 0.0
                        self.data.qfrc_applied[self.v_adr] = tau
                        self.data.qfrc_applied[self.neck_v_adr] = neck_tau
                        applied_viewer_force = self._begin_viewer_push_substep()
                        self._apply_viewer_push_substep(applied_viewer_force)
                        try:
                            mujoco.mj_step(self.model, self.data)
                        finally:
                            self._end_viewer_push_substep(applied_viewer_force)
                    v.opt.geomgroup[:] = geomgroup0
                    v.sync()
                    dt_left = self.control_dt - (time.time() - t0)
                    if dt_left > 0:
                        time.sleep(dt_left)
        finally:
            self._stdin_alive = False
            restore_stdin()
            if self.run_logger:
                self.run_logger.close()
            if self.bridge:
                self.bridge.stop()
            if self.gamepad:
                self.gamepad.stop()
            if self.real_panel:
                self.real_panel.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config/oceanbdx.yaml"))
    ap.add_argument("--policy", default=None, help="覆盖config中的行走policy路径")
    ap.add_argument("--stand-policy", default=None,
                    help="覆盖config中的站立policy路径 (policy.stand_path)")
    ap.add_argument("--gamepad", action="store_true",
                    help="启用论文附录C的USB手柄站立/行走模式与命令映射")
    ap.add_argument("--no-policy", action="store_true", help="仅验证起立脚本, 不加载策略")
    ap.add_argument("--real", action="store_true",
                    help="连接真机电机: 普通sim2sim中输出仿真目标角; 与 --manual 组合时为滑块联调")
    ap.add_argument("--manual", action="store_true",
                    help="打开关节角拖动滑块面板 (可与 --real 组合)")
    ap.add_argument("--no-log", action="store_true",
                    help="关闭 runlog/ 观测+动作记录与终端刷新 (默认开启)")
    ap.add_argument("--debug-obs", action="store_true",
                    help="打印base姿态/IMU等效观测/policy动作, 用于检查sim2sim坐标系")
    ap.add_argument("--debug-actions", action="store_true",
                    help="配合--debug-obs打印每个关节的raw action/action/target/q/dq")
    ap.add_argument("--debug-actions-full", action="store_true",
                    help="配合--debug-actions打印完整raw action/action/target/q/dq数组")
    ap.add_argument("--debug-balance", action="store_true",
                    help="配合--debug-obs打印base倾斜/横移和左右脚接触力")
    ap.add_argument("--debug-obs-interval", type=float, default=0.2,
                    help="--debug-obs的打印间隔(s), 默认0.2")
    ap.add_argument("--probe-policy", action="store_true",
                    help="不启动viewer, 用理想零位直立观测直接测试ONNX策略输出")
    ap.add_argument("--debug-push-steps", type=int, default=0,
                    help="不启动viewer, 自动起立后在RL_STAND中运行固定外力测试, 单位为policy step")
    ap.add_argument("--debug-push-start", type=int, default=80,
                    help="固定外力开始的policy step, 从进入RL_STAND后计数")
    ap.add_argument("--debug-push-duration", type=int, default=11,
                    help="固定外力持续的policy step")
    ap.add_argument("--debug-push-force-x", type=float, default=0.0,
                    help="固定外力世界系X方向[N]")
    ap.add_argument("--debug-push-force-y", type=float, default=40.0,
                    help="固定外力世界系Y方向[N]")
    ap.add_argument("--viewer-push-force-x", type=float, default=0.0,
                    help="运行终端短按5施加到base_link的世界系X方向力[N]，短按6取反")
    ap.add_argument("--viewer-push-force-y", type=float, default=0.0,
                    help="运行终端短按5施加到base_link的世界系Y方向力[N]，短按6取反")
    ap.add_argument("--viewer-push-force-z", type=float, default=0.0,
                    help="运行终端短按5施加到base_link的世界系Z方向力[N]，短按6取反")
    ap.add_argument("--viewer-push-duration", type=float, default=0.1,
                    help="可视化定量推力持续时间[s]，默认0.1以匹配Table V大推")
    ap.add_argument("--sim-action-scale", type=float, default=None,
                    help="仅本次sim2sim运行覆盖policy action_scale")
    ap.add_argument("--sim-rl-kd", type=float, default=None,
                    help="仅本次sim2sim运行覆盖所有腿部RL kd")
    ap.add_argument("--sim-rl-kd-list", default=None,
                    help="仅本次sim2sim运行覆盖逐关节RL kd, 10个逗号分隔数值")
    args = ap.parse_args()

    if not os.path.exists(SCENE_XML):
        print("scene xml missing, run: python3 scripts/urdf2mjcf.py")
        sys.exit(1)
    sim = Sim(args)
    sim.probe_policy_only = args.probe_policy
    sim.run()
