#!/usr/bin/env python3
"""
OceanBDX sim2sim: MuJoCo + ONNX policy 验证

复刻真实机器人的状态机最小闭环: SIT -> STAND_UP(脚本插值) -> RL_STAND(站立模型) <-> RL_WALK(行走模型)
★ 站立与行走是两个独立训练/导出的 ONNX 模型 (policy/stand/policy.onnx 与 policy/policy.onnx)：
  - 起立完成后自动进入 RL_STAND, 加载站立模型 (74 维观测, torso 位姿命令, 无相位);
  - 按 2 切到行走模型 (RL_WALK, 77 维观测, 速度命令 + 相位); 按 1 切回站立模型。
观测/动作处理与 C++ 部署代码 (src/policy.cpp) 完全一致, 用于在上真机前验证:
  - IsaacLab 导出的两个 policy.onnx 正确性
  - 观测顺序/缩放/默认关节角配置正确性
  - 起立脚本与RL切换的衔接、站立/行走模型互切

用法 (在 oceanbdx 根目录):
    python3 sim2sim/mujoco_sim.py [--policy policy/policy.onnx] [--config config/oceanbdx.yaml]
    python3 sim2sim/mujoco_sim.py --no-policy     # 无策略, 只验证起立脚本+站立PD保持
    python3 sim2sim/mujoco_sim.py --manual        # 纯sim: 拖动滑块摆关节角
    python3 sim2sim/mujoco_sim.py --real --no-policy # 仿真目标角→真机, 面板显示真机误差
    python3 sim2sim/mujoco_sim.py --real --manual # 真机联调: 拖动滑块→PD下发到真机
    # ★ --real 需先安装 unitree_actuator_sdk, 且停掉 C++ 主控 oceanbdx_run (串口互斥)
    # 默认每次 policy 推理把 观测+raw/clip动作+目标角 写入 runlog/sim2sim_<时间戳>.csv,
    # 并在终端刷新 state/cmd/高度/倾斜/速度/|act|max/饱和数; 用 --no-log 关闭。

键盘 (★推荐聚焦“终端窗口”操作, 与真机 main.cpp 一致, 不触发 MuJoCo 快捷键):
    0 = 真机缓慢到蹲姿   1 = 起立/切站立模型   2 = 切行走模型   3 = 切站立模型(同1)   9 = 阻尼   r = 重置
    p = 真机电机输出开关
    w/s = vx±0.1   a/d = vy±0.1   q/e = wz±0.1   x = 速度清零
    (速度指令仅在 RL_WALK 行走模型生效, 先按 2 进入行走)
    t/g = 躯干pitch±   v/c = 躯干yaw±   y/b = 躯干roll±   f/z = 躯干高度±
    (躯干位姿命令仅在 RL_STAND 站立模型生效, 先按 1 进入站立)
    MuJoCo 窗口内也可按同样的键(方向键映射到 w/s/q/e), 但数字键会附带触发
    MuJoCo 自带 geomgroup 切换, 已每帧复位防止机器人 geom 被隐藏。
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(ROOT, "sim2sim/ocean_scene.xml")

LEG_JOINTS = ["leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
              "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint"]
NECK_JOINTS = ["neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint"]
# 动作向量顺序 = 腿 10 + 脖子 4（与训练侧 action 布局一致），调试打印按此索引
ACTION_JOINTS = LEG_JOINTS + NECK_JOINTS


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


def wrap_angle(a):
    """把角度环绕到 (-π, π]。与训练侧 path_frame.wrap_angle 一致。"""
    return np.arctan2(np.sin(a), np.cos(a))


class PathFrameNP:
    """path frame（BDX 论文 V-A / Fig.4）的单 env numpy 复刻。

    逐行为对齐训练侧 oceanisaaclab .../path_frame.py：+x 轴=头部前向，行走按 path 系
    命令积分、站立一阶低通收敛到双脚中心+躯干朝向、最大偏差投影拉回躯干附近。
    sim2sim 用 MuJoCo 真值驱动（躯干世界 xy/yaw 里程计 + 双脚 FK 中心）；真机侧改由
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

    def step(self, dt, cmd, moving, base_pos_xy, head_yaw, feet_center_xy):
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        if moving:
            # 行走：path 系命令 (vx 头前, vy 头左, wz) 旋到世界系再积分
            dx_w = cmd[0] * cos_y - cmd[1] * sin_y
            dy_w = cmd[0] * sin_y + cmd[1] * cos_y
            self.pos = self.pos + np.array([dx_w, dy_w]) * dt
            self.yaw = self.yaw + cmd[2] * dt
        else:
            # 站立：一阶低通收敛到双脚中心 + 躯干朝向
            alpha = min(1.0, dt / self.stand_time_constant)
            self.pos = self.pos + alpha * (np.asarray(feet_center_xy, dtype=float) - self.pos)
            self.yaw = self.yaw + alpha * wrap_angle(head_yaw - self.yaw)
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
    """路线 B（BDX 论文复刻）77 维观测构造与逐关节动作解码（阶段1：含脖子+头部命令）。

    须与训练侧 oceanisaaclab .../oceanisaaclab_walk_env.py `_get_observations` 逐位一致：
      [0:2]   pos_pf × pos_pf_scale                path 系躯干 xy
      [2:4]   (sin, cos) 相对 yaw = head_yaw − path_yaw
      [4:7]   lin_vel_b × lin_vel_scale            body 系线速度
      [7:10]  ang_vel_b × ang_vel_scale            body 系角速度（= gyro）
      [10:20] (q_leg − default) × dof_pos_scale
      [20:24] q_neck × dof_pos_scale               脖子 4 关节角（无 default 偏移）
      [24:34] dq_leg × dof_vel_scale
      [34:38] dq_neck × dof_vel_scale
      [38:52] a_{t-1}（14：10 腿 + 4 脖子）
      [52:66] a_{t-2}
      [66:70] (sin2πφ, cos2πφ, sin4πφ, cos4πφ)     相位二阶谐波
      [70:73] cmd × commands_scale
      [73:77] head_cmd × head_command_scale        (Δh, pitch, yaw, roll)
    动作解码（14 维）：前 10 腿 target = default + action_joint_ranges⊙clip；
      后 4 脖子 target = neck_default + neck_action_joint_ranges⊙clip。
    path frame 状态机（PathFrameNP）由外部用 MuJoCo 真值逐控制步 step() 推进。
    """

    def __init__(self, onnx_path, cfg, num_joints, num_neck=4, stand=False):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.cfg = cfg
        self.nj = num_joints  # 腿关节数（10）
        self.n_neck = num_neck  # 脖子关节数（4）
        self.n_act = num_joints + num_neck  # 动作维度（14）
        # stand=True：独立训练的站立（perpetual）模型。观测去相位谐波、命令由
        #   (cmd3 行走速度) 换成 (torso4 躯干位姿命令 h,pitch,yaw,roll)，共 74 维；
        #   动作解码与行走完全一致。stand=False：行走模型，77 维（含 phase4 + cmd3）。
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
        # 相位速率 φ̇：恒定步频库即常数 1/period（与训练 sample_phase_rate 在本库上等价）
        self.phase_rate = 1.0 / max(1.0e-6, self.gait_cycle_period)
        self.gait_phase = 0.0
        self.last_actions = np.zeros(self.n_act, dtype=np.float32)
        self.last_last_actions = np.zeros(self.n_act, dtype=np.float32)
        self.last_obs = None
        self.last_projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.last_raw_actions = np.zeros(self.n_act, dtype=np.float32)

    def reset(self):
        self.last_actions[:] = 0
        self.last_last_actions[:] = 0
        self.last_obs = None
        self.last_projected_gravity[:] = [0.0, 0.0, -1.0]
        self.last_raw_actions[:] = 0
        self.gait_phase = 0.0

    def step(self, q, dq, gyro, cmd, pos_pf, yaw_pf, lin_vel_b,
             neck_q, neck_dq, head_cmd, torso_cmd=None):
        """一次策略推理（行走 77 维 / 站立 74 维，由 self.stand 决定观测尾部）。

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

        # 观测公共块（行走/站立一致，66 维）：path 系位姿 + 速度 + 关节 + 双帧动作历史。
        obs_common = [
            np.asarray(pos_pf, dtype=np.float32) * self.pos_pf_scale,
            yaw_feat,
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
            # 站立尾部：torso_cmd4 + head_cmd4（无相位谐波，共 74 维）。
            if torso_cmd is None:
                torso_cmd = np.zeros(4, dtype=np.float32)
            obs_tail = [
                np.asarray(torso_cmd, dtype=np.float32) * self.torso_command_scale,
                head_cmd * self.head_command_scale,
            ]
        else:
            # 行走尾部：phase 二阶谐波4 + cmd3 + head_cmd4（共 77 维）。
            # 相位积分：φ̇ 恒定步频；站立不清零（训练 obs 不对站立屏蔽谐波）。
            # 训练侧 _pre_physics_step 在 _get_observations 之前推进相位，obs 见已推进的 φ。
            self.gait_phase = (self.gait_phase + self.phase_rate * self.policy_dt) % 1.0
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
    """把每个 policy step 的观测/动作/姿态写入带时间戳的 CSV (runlog/),
    并按节流间隔在终端刷新关键参数, 便于实时观察策略是否生效。

    每行 = 一次 policy 推理 (RL_BALANCE / RL_WALK 下 decimation 对齐时)。
    观测拆成各物理量分列, 动作记录 raw(网络原始) / clip(裁剪后) / target(下发关节角)。
    """

    def __init__(self, nj, term_interval=0.3, path=None):
        self.nj = nj
        self.term_interval = float(term_interval)
        self._last_term = -1e9
        os.makedirs(os.path.join(ROOT, "runlog"), exist_ok=True)
        if path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(ROOT, "runlog", f"sim2sim_{stamp}.csv")
        self.path = path
        self.f = open(path, "w", buffering=1)  # 行缓冲, 实时落盘
        leg = [j.replace("_joint", "") for j in LEG_JOINTS]
        cols = (["t", "state", "cmd_vx", "cmd_vy", "cmd_wz",
                 "base_z", "tilt_x", "tilt_y",
                 "vel_bx", "vel_by", "vel_bz",
                 "gyro_x", "gyro_y", "gyro_z"]
                + [f"q_{j}" for j in leg]
                + [f"dq_{j}" for j in leg]
                + [f"raw_{j}" for j in leg]
                + [f"act_{j}" for j in leg]
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

        if t - self._last_term >= self.term_interval:
            self._last_term = t
            sat = int(np.sum(np.abs(action) > 0.98))
            print(
                f"[run] t={t:6.2f} {state:10s} "
                f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}] "
                f"z={base_z:.3f} tilt=[{tilt[0]:+.3f},{tilt[1]:+.3f}] "
                f"vel_b=[{vel_b[0]:+.2f},{vel_b[1]:+.2f}] "
                f"|act|max={np.max(np.abs(action)):.2f} sat={sat}/{self.nj} "
                f"|dq|max={np.max(np.abs(dq)):.2f}"
            )

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class Sim:
    def __init__(self, args):
        full = yaml.safe_load(open(args.config))["oceanbdx"]
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
        self.damping_kd = ctrl["damping_kd"]

        self.joint_lower = np.array(ctrl["joint_lower"][:self.nj], dtype=float)
        self.joint_upper = np.array(ctrl["joint_upper"][:self.nj], dtype=float)

        cal = full["calibration"]
        self.sit_pose = np.array(cal["sit_pose"][:self.nj], dtype=float)
        self.stand_pose = np.array(cal["stand_pose"][:self.nj], dtype=float)

        cmd_cfg = full["command"]
        self.max_vel = np.array([cmd_cfg["max_vx"], cmd_cfg["max_vy"], cmd_cfg["max_wz"]])

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

        # 两个独立模型：行走 (RL_WALK, 77 维) 与站立 (RL_STAND, 74 维)。
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
        # 双脚 geom（站立收敛目标 = 双脚中心 FK 真值）
        self.foot_geom_ids = [self.model.geom("foot_r").id, self.model.geom("foot_l").id]

        # ---- 2026-07-08 脖子/头部命令 ----
        self.n_neck = len(NECK_JOINTS)
        self.head_cmd = np.zeros(4, dtype=np.float32)  # (Δh, pitch, yaw, roll)
        # 站立躯干命令 (h, pitch, yaw, roll)，仅 RL_STAND 生效；默认全 0=标称直立站姿。
        # 键盘微调见 _cmd_key：t/g=pitch± v/c=yaw± y/b=roll± f/z=高度±（切状态清零）。
        self.torso_cmd = np.zeros(4, dtype=np.float32)
        # 站立躯干命令范围（对齐训练侧 torso_command_*_range，超范围会被钳到边界）
        self.max_torso = np.array([0.05, 0.25, 0.35, 0.18], dtype=np.float32)
        cmd_cfg = full.get("command", {})
        self.max_head = np.array([
            float(cmd_cfg.get("max_head_dh", 0.02)),
            float(cmd_cfg.get("max_head_pitch", 0.5)),
            float(cmd_cfg.get("max_head_yaw", 1.0)),
            float(cmd_cfg.get("max_head_roll", 0.6)),
        ], dtype=np.float32)
        self.neck_kp = float(pcfg_full.get("neck_kp", 50.0))
        self.neck_kd = float(pcfg_full.get("neck_kd", 2.0))
        neck_default = pcfg_full.get("neck_default_dof_pos", [0.0] * self.n_neck)
        self.neck_default = np.array(neck_default, dtype=np.float32)
        # 脖子位置目标（策略输出，RL 前锁默认位）
        self.neck_target = self.neck_default.copy()

        self.state = "SIT"
        self.state_time = 0.0
        self.cmd = np.zeros(3)
        self.run_logger = None
        self.want_runlog = not bool(getattr(args, "no_log", False))
        self.rl_target = self.stand_pose.copy()
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
        # 理想直立零命令：path 系躯干位于原点、朝向对齐、线速度 0
        pos_pf = np.zeros(2, dtype=np.float32)
        yaw_pf = 0.0
        lin_vel_b = np.zeros(3, dtype=np.float32)
        neck_q = np.zeros(self.n_neck, dtype=np.float32)
        neck_dq = np.zeros(self.n_neck, dtype=np.float32)
        head_cmd = np.zeros(4, dtype=np.float32)
        target, _neck_target = self.policy.step(
            q, dq, gyro, cmd, pos_pf, yaw_pf, lin_vel_b, neck_q, neck_dq, head_cmd
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
        self.cmd[:] = 0
        self.rl_target = self.stand_pose.copy()
        self.last_target = self.sit_pose.copy()
        self.last_kp = self.fixed_kp.copy()
        self.last_kd = self.fixed_kd.copy()
        self.neck_target = self.neck_default.copy()
        self.head_cmd[:] = 0.0
        self.torso_cmd[:] = 0.0
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
        """从 MuJoCo 读 path frame 所需真值：躯干 xy、头部 yaw、双脚中心 xy、body 线速度。

        真机侧这些量改由状态估计器/里程计 + FK 提供（论文 V-D 要求 runtime 一致）。
        """
        base_xy = self.data.qpos[0:2].copy()
        head_yaw = yaw_from_quat(self.data.qpos[3:7]) + self.head_yaw_offset
        feet_center = np.mean(
            [self.data.geom_xpos[gid, :2] for gid in self.foot_geom_ids], axis=0
        )
        _, lin_vel_b = self.base_linear_velocities()
        return base_xy, head_yaw, feet_center, lin_vel_b

    def foot_contact_forces(self):
        left_force = 0.0
        right_force = 0.0
        try:
            left_gid = self.model.geom("foot_l").id
            right_gid = self.model.geom("foot_r").id
        except KeyError:
            return left_force, right_force
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 not in (left_gid, right_gid) and contact.geom2 not in (left_gid, right_gid):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = max(0.0, float(force[0]))
            if contact.geom1 == left_gid or contact.geom2 == left_gid:
                left_force += normal_force
            if contact.geom1 == right_gid or contact.geom2 == right_gid:
                right_force += normal_force
        return left_force, right_force

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

    def _cmd_key(self, c):
        """处理单个按键字符 (已转小写)。窗口回调与终端线程共用。

        与真机 src/main.cpp 一致: 0/1/2/3=状态 9=阻尼 r=重置 p=电机开关
        w/s=vx± a/d=vy± q/e=wz± x=速度清零
        """
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
        elif c == "9":
            self.state = "DAMPING"
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
        elif c == "i":
            self.head_cmd[1] = min(self.head_cmd[1] + 0.1, self.max_head[1])
        elif c == "k":
            self.head_cmd[1] = max(self.head_cmd[1] - 0.1, -self.max_head[1])
        elif c == "j":
            self.head_cmd[2] = min(self.head_cmd[2] + 0.1, self.max_head[2])
        elif c == "l":
            self.head_cmd[2] = max(self.head_cmd[2] - 0.1, -self.max_head[2])
        elif c == "u":
            self.head_cmd[3] = min(self.head_cmd[3] + 0.1, self.max_head[3])
        elif c == "o":
            self.head_cmd[3] = max(self.head_cmd[3] - 0.1, -self.max_head[3])
        elif c == "n":
            self.head_cmd[0] = min(self.head_cmd[0] + 0.005, self.max_head[0])
        elif c == "m":
            self.head_cmd[0] = max(self.head_cmd[0] - 0.005, -self.max_head[0])
        elif c == "h":
            self.head_cmd[:] = 0.0
        # ---- 站立躯干命令（仅 RL_STAND 生效，微调站姿）----
        # t/g=前后倾pitch± v/c=偏航yaw± y/b=侧倾roll± f/z=高度h±
        elif c == "t":
            self.torso_cmd[1] = min(self.torso_cmd[1] + 0.05, self.max_torso[1])
        elif c == "g":
            self.torso_cmd[1] = max(self.torso_cmd[1] - 0.05, -self.max_torso[1])
        elif c == "v":
            self.torso_cmd[2] = min(self.torso_cmd[2] + 0.05, self.max_torso[2])
        elif c == "c":
            self.torso_cmd[2] = max(self.torso_cmd[2] - 0.05, -self.max_torso[2])
        elif c == "y":
            self.torso_cmd[3] = min(self.torso_cmd[3] + 0.05, self.max_torso[3])
        elif c == "b":
            self.torso_cmd[3] = max(self.torso_cmd[3] - 0.05, -self.max_torso[3])
        elif c == "f":
            self.torso_cmd[0] = min(self.torso_cmd[0] + 0.005, self.max_torso[0])
        elif c == "z":
            self.torso_cmd[0] = max(self.torso_cmd[0] - 0.005, -self.max_torso[0])
        else:
            return
        if c in "wsadqex":
            tag = "" if self.state == "RL_WALK" else "  (仅 RL_WALK 生效, 先按2)"
            print(f"cmd = vx={self.cmd[0]:+.2f} vy={self.cmd[1]:+.2f} wz={self.cmd[2]:+.2f}{tag}")
        if c in "ikjluonmh":
            print(f"head_cmd = Δh={self.head_cmd[0]:+.3f} pitch={self.head_cmd[1]:+.2f} "
                  f"yaw={self.head_cmd[2]:+.2f} roll={self.head_cmd[3]:+.2f}")
        if c in "tgvcybfz":
            tag = "" if self.state == "RL_STAND" else "  (仅 RL_STAND 生效, 先按1)"
            print(f"torso_cmd = h={self.torso_cmd[0]:+.3f} pitch={self.torso_cmd[1]:+.2f} "
                  f"yaw={self.torso_cmd[2]:+.2f} roll={self.torso_cmd[3]:+.2f}{tag}")

    def _switch_rl_state(self, target):
        """在 RL_STAND(站立模型) 与 RL_WALK(行走模型) 之间切换。

        两个模型互相独立：切换时把目标模型的动作历史/相位清零(fresh start)，
        目标关节角先锁到当前腿角避免 PD 跳变，命令清零，rl_tick 归零使下一控制步
        立即触发一次目标模型推理。path frame 保持连续(仍由 MuJoCo 真值驱动)。
        """
        if self.state == target:
            return
        self.state = target
        if target == "RL_WALK":
            self.policy = self.walk_policy
            self.cmd[:] = 0.0  # 切入行走时速度命令归零，等待手动加速
            label = "walk model"
        else:  # RL_STAND
            self.policy = self.stand_policy or self.walk_policy
            self.torso_cmd[:] = 0.0  # 切入站立时躯干命令归零(标称直立)
            label = "stand model" if self.stand_policy else "stand(回退行走模型)"
        if self.policy is not None:
            self.policy.reset()
        self.rl_target = self.data.qpos[self.q_adr].copy()
        self.rl_tick = 0
        print(f"[FSM] -> {target} ({label})")

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
                print("[FSM] attitude protect -> DAMPING")

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
                self.rl_tick = 0
                if self.policy:
                    self.policy.reset()
                # path frame 初始化到当前躯干出生位姿（论文：起始于躯干状态）
                base_xy, head_yaw, _, _ = self.path_frame_truth()
                self.path_frame.reset(base_xy, head_yaw)
                self.state, self.state_time = "RL_STAND", 0.0
                label = "stand model" if self.stand_policy else "stand(回退行走模型)"
                print(f"[FSM] STAND_UP -> RL_STAND ({label})")
        elif self.state in ("RL_STAND", "RL_WALK"):
            # 按状态选活动模型：RL_WALK→行走模型(cmd 速度)，RL_STAND→站立模型(torso 命令)
            if self.state == "RL_WALK":
                self.policy = self.walk_policy
                cmd = self.cmd
                torso_cmd = np.zeros(4, dtype=np.float32)
            else:  # RL_STAND
                self.policy = self.stand_policy or self.walk_policy
                cmd = np.zeros(3)
                torso_cmd = self.torso_cmd
            if self.policy is None:
                target, kp, kd = self.rl_target, self.kp, self.kd  # 无策略兜底: 锁站姿
            else:
                fresh_policy = self.rl_tick % self.decimation == 0
                if fresh_policy:
                    # path frame 每策略步推进一次（用 MuJoCo 真值；真机侧改状态估计器）
                    base_xy, head_yaw, feet_center, lin_vel_b = self.path_frame_truth()
                    # 只有行走且命令超阈值才平移 path frame；站立走收敛分支
                    moving = self.state == "RL_WALK" and (
                        float(np.max(np.abs(cmd[:2]))) > self.move_command_threshold
                        or abs(float(cmd[2])) > self.move_command_threshold
                    )
                    self.path_frame.step(
                        self.policy.policy_dt, cmd, moving, base_xy, head_yaw, feet_center
                    )
                    pos_pf, yaw_pf = self.path_frame.base_in_path_frame(base_xy, head_yaw)
                    neck_q = self.data.qpos[self.neck_q_adr].copy()
                    neck_dq = self.data.qvel[self.neck_v_adr].copy()
                    self.rl_target, self.neck_target = self.policy.step(
                        q, dq, gyro, cmd, pos_pf, yaw_pf, lin_vel_b,
                        neck_q, neck_dq, self.head_cmd, torso_cmd,
                    )
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
        else:  # DAMPING
            target, kp, kd = q, np.zeros(self.nj), np.full(self.nj, self.damping_kd)

        self.debug_print_obs(q, dq, quat, gyro, target)

        self.last_target = target.copy()
        self.last_kp = kp.copy()
        self.last_kd = kd.copy()

        tau = kp * (target - q) - kd * dq
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        # 脖子: RL 状态下位置伺服跟随策略输出的脖子目标（阶段1 起脖子参与控制），
        # 其余状态锁默认位。增益对齐训练侧 neck ImplicitActuator（kp50/kd2）。
        neck_q = self.data.qpos[self.neck_q_adr]
        neck_dq = self.data.qvel[self.neck_v_adr]
        if self.state in ("RL_STAND", "RL_WALK"):
            neck_ref = self.neck_target
        else:
            neck_ref = self.neck_default
        neck_tau = self.neck_kp * (neck_ref - neck_q) - self.neck_kd * neck_dq

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
        print(__doc__)

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
                # 记录初始 geom 可见组, 每帧复位; 防止窗口内误按数字键
                # 触发 MuJoCo 自带 geomgroup 切换把机器人 geom 隐藏。
                geomgroup0 = np.array(v.opt.geomgroup).copy()
                while v.is_running() and (self.real_panel is None or self.real_panel.alive()):
                    t0 = time.time()
                    tau, neck_tau = self.control_step()
                    self.update_real_output()
                    for _ in range(sim_steps_per_ctrl):
                        self.data.qfrc_applied[self.v_adr] = tau
                        self.data.qfrc_applied[self.neck_v_adr] = neck_tau
                        mujoco.mj_step(self.model, self.data)
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
            if self.real_panel:
                self.real_panel.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config/oceanbdx.yaml"))
    ap.add_argument("--policy", default=None, help="覆盖config中的行走policy路径")
    ap.add_argument("--stand-policy", default=None,
                    help="覆盖config中的站立policy路径 (policy.stand_path)")
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
                    help="不启动viewer, 自动起立后在RL_BALANCE中运行固定外力测试, 单位为policy step")
    ap.add_argument("--debug-push-start", type=int, default=80,
                    help="固定外力开始的policy step, 从进入RL_BALANCE后计数")
    ap.add_argument("--debug-push-duration", type=int, default=11,
                    help="固定外力持续的policy step")
    ap.add_argument("--debug-push-force-x", type=float, default=0.0,
                    help="固定外力世界系X方向[N]")
    ap.add_argument("--debug-push-force-y", type=float, default=40.0,
                    help="固定外力世界系Y方向[N]")
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
