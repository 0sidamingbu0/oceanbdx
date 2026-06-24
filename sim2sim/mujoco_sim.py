#!/usr/bin/env python3
"""
OceanBDX sim2sim: MuJoCo + ONNX policy 验证

复刻真实机器人的状态机最小闭环: SIT -> STAND_UP(脚本插值) -> RL_BALANCE -> RL_WALK
观测/动作处理与 C++ 部署代码 (src/policy.cpp) 完全一致, 用于在上真机前验证:
  - IsaacLab 导出的 policy.onnx 正确性
  - 观测顺序/缩放/默认关节角配置正确性
  - 起立脚本与RL切换的衔接

用法 (在 oceanbdx 根目录):
    python3 sim2sim/mujoco_sim.py [--policy policy/policy.onnx] [--config config/oceanbdx.yaml]
    python3 sim2sim/mujoco_sim.py --no-policy     # 无策略, 只验证起立脚本+站立PD保持
    python3 sim2sim/mujoco_sim.py --manual        # 纯sim: 拖动滑块摆关节角
    python3 sim2sim/mujoco_sim.py --real --no-policy # 仿真目标角→真机, 面板显示真机误差
    python3 sim2sim/mujoco_sim.py --real --manual # 真机联调: 拖动滑块→PD下发到真机
    # ★ --real 需先安装 unitree_actuator_sdk, 且停掉 C++ 主控 oceanbdx_run (串口互斥)

键盘 (MuJoCo viewer 窗口内):
    0 = 真机缓慢到蹲姿   1 = 起立   2 = 行走   3 = 回平衡   9 = 阻尼   r = 重置
    p = 真机电机输出开关
    ↑/↓ = vx±0.1   ←/→ = wz±0.1   x = 速度清零
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


def quat_rotate_inverse_gravity(q):
    """projected gravity = quat_rotate_inverse(q, (0,0,-1)), q=(w,x,y,z). 与 policy.cpp 一致"""
    w, x, y, z = q
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ])


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
    """与 src/policy.cpp 相同的观测构造与动作解码"""

    def __init__(self, onnx_path, cfg, num_joints):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.cfg = cfg
        self.nj = num_joints
        self.default_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float32)
        self.commands_scale = np.array(cfg.get("commands_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        self.last_actions = np.zeros(self.nj, dtype=np.float32)
        self.last_obs = None
        self.last_projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.last_raw_actions = np.zeros(self.nj, dtype=np.float32)

    def reset(self):
        self.last_actions[:] = 0
        self.last_obs = None
        self.last_projected_gravity[:] = [0.0, 0.0, -1.0]
        self.last_raw_actions[:] = 0

    def step(self, q, dq, quat, gyro, cmd):
        c = self.cfg
        projected_gravity = quat_rotate_inverse_gravity(quat)
        obs = np.concatenate([
            gyro * c["ang_vel_scale"],
            projected_gravity,
            np.asarray(cmd) * self.commands_scale,
            (q - self.default_dof_pos) * c["dof_pos_scale"],
            dq * c["dof_vel_scale"],
            self.last_actions,
        ]).astype(np.float32)
        self.last_projected_gravity = projected_gravity.astype(np.float32)
        obs = np.clip(obs, -c["clip_obs"], c["clip_obs"])
        self.last_obs = obs.copy()
        act = self.sess.run(None, {self.input_name: obs[None, :]})[0][0]
        self.last_raw_actions = act.astype(np.float32)
        act = np.clip(act, -c["clip_actions"], c["clip_actions"])
        self.last_actions = act.astype(np.float32)
        return self.default_dof_pos + c["action_scale"] * act


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

        self.policy = None
        if not args.no_policy:
            pol_path = args.policy or os.path.join(ROOT, full["policy"]["path"])
            if os.path.exists(pol_path):
                pcfg = dict(full["policy"])
                pcfg["default_dof_pos"] = pcfg["default_dof_pos"][:self.nj]
                sim2sim_ctrl = full.get("sim2sim", {})
                if "action_scale" in sim2sim_ctrl:
                    pcfg["action_scale"] = sim2sim_ctrl["action_scale"]
                if getattr(args, "sim_action_scale", None) is not None:
                    pcfg["action_scale"] = float(args.sim_action_scale)
                self.policy = Policy(pol_path, pcfg, self.nj)
                print(f"[sim2sim] policy loaded: {pol_path}")
            else:
                print(f"[sim2sim] policy not found at {pol_path}, running script-only mode")

        self.state = "SIT"
        self.state_time = 0.0
        self.cmd = np.zeros(3)
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
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gyro = np.zeros(3, dtype=np.float32)
        cmd = np.zeros(3, dtype=np.float32)
        target = self.policy.step(q, dq, quat, gyro, cmd)
        action = self.policy.last_actions
        raw_action = self.policy.last_raw_actions
        saturated = [LEG_JOINTS[i] for i, value in enumerate(np.abs(action)) if value > 0.98]
        active = np.argsort(-np.abs(action))[:4]
        active_text = ", ".join(f"{LEG_JOINTS[i]}={action[i]:+.3f}" for i in active)
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
        if self.policy:
            self.policy.reset()
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
            raw_action = np.zeros(self.nj)
            action = np.zeros(self.nj)
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
                f"{LEG_JOINTS[i]}={action[i]:+.3f}" for i in active
            )
            saturated = [LEG_JOINTS[i] for i, value in enumerate(np.abs(action)) if value > 0.98]
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
            in_rl = self.state in ("RL_BALANCE", "RL_WALK")
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
        c = chr(keycode) if keycode < 256 else ""
        if c == "0":
            self.start_real_sit_align()
        elif c in ("p", "P"):
            if not self.bridge:
                print("[real] 未连接真机桥接, 忽略电机输出开关")
                return
            self.real_output_enabled = not self.real_output_enabled
            if self.real_panel:
                self.real_panel.set_enabled(self.real_output_enabled)
            if self.real_output_enabled:
                self.real_cmd_q = self.bridge.get_state()[0][:self.nj].copy()
            print(f"[real] motor output {'ENABLED' if self.real_output_enabled else 'DISABLED'}")
        elif c == "1" and self.state == "SIT":
            if self.real_sit_active:
                print("[real] 真机仍在缓慢回蹲姿, 完成后再按 1")
                return
            if self.bridge and not self.real_sit_done:
                print("[real] 先按 0 让真机缓慢回蹲姿; 到位后再按 1")
                return
            self.stand_start = self.data.qpos[self.q_adr].copy()
            self.state, self.state_time = "STAND_UP", 0.0
            print("[FSM] SIT -> STAND_UP")
        elif c == "2" and self.state == "RL_BALANCE":
            self.state = "RL_WALK"
            print("[FSM] RL_BALANCE -> RL_WALK")
        elif c == "3" and self.state == "RL_WALK":
            self.state = "RL_BALANCE"
            print("[FSM] RL_WALK -> RL_BALANCE")
        elif c == "9":
            self.state = "DAMPING"
            self.disable_real_output("DAMPING")
            print("[FSM] -> DAMPING")
        elif c in ("r", "R"):
            self.reset()
            self.disable_real_output("RESET")
        elif c == "x":
            self.cmd[:] = 0
        elif keycode == 265:  # up
            self.cmd[0] = min(self.cmd[0] + 0.1, self.max_vel[0])
        elif keycode == 264:  # down
            self.cmd[0] = max(self.cmd[0] - 0.1, -self.max_vel[0])
        elif keycode == 263:  # left
            self.cmd[2] = min(self.cmd[2] + 0.1, self.max_vel[2])
        elif keycode == 262:  # right
            self.cmd[2] = max(self.cmd[2] - 0.1, -self.max_vel[2])
        if c in "0123pPrR":
            return
        print(f"cmd = {self.cmd}")

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
            if self.state in ("SIT", "STAND_UP", "RL_BALANCE", "RL_WALK"):
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
        if self.state in ("STAND_UP", "RL_BALANCE", "RL_WALK"):
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
                if self.policy:
                    self.policy.reset()
                    self.rl_target = self.stand_pose.copy()
                    self.rl_tick = 0
                    self.state, self.state_time = "RL_BALANCE", 0.0
                    print("[FSM] STAND_UP -> RL_BALANCE")
        elif self.state in ("RL_BALANCE", "RL_WALK"):
            cmd = self.cmd if self.state == "RL_WALK" else np.zeros(3)
            if self.rl_tick % self.decimation == 0:
                self.rl_target = self.policy.step(q, dq, quat, gyro, cmd)
            self.rl_tick += 1
            target, kp, kd = self.rl_target, self.kp, self.kd
        else:  # DAMPING
            target, kp, kd = q, np.zeros(self.nj), np.full(self.nj, self.damping_kd)

        self.debug_print_obs(q, dq, quat, gyro, target)

        self.last_target = target.copy()
        self.last_kp = kp.copy()
        self.last_kd = kd.copy()

        tau = kp * (target - q) - kd * dq
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        # 脖子: 锁定零位 (脖子暂不参与控制)
        neck_q = self.data.qpos[self.neck_q_adr]
        neck_dq = self.data.qvel[self.neck_v_adr]
        neck_tau = 5.0 * (0.0 - neck_q) - 0.5 * neck_dq

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

        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_cb) as v:
                while v.is_running() and (self.real_panel is None or self.real_panel.alive()):
                    t0 = time.time()
                    tau, neck_tau = self.control_step()
                    self.update_real_output()
                    for _ in range(sim_steps_per_ctrl):
                        self.data.qfrc_applied[self.v_adr] = tau
                        self.data.qfrc_applied[self.neck_v_adr] = neck_tau
                        mujoco.mj_step(self.model, self.data)
                    v.sync()
                    dt_left = self.control_dt - (time.time() - t0)
                    if dt_left > 0:
                        time.sleep(dt_left)
        finally:
            if self.bridge:
                self.bridge.stop()
            if self.real_panel:
                self.real_panel.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config/oceanbdx.yaml"))
    ap.add_argument("--policy", default=None, help="覆盖config中的policy路径")
    ap.add_argument("--no-policy", action="store_true", help="仅验证起立脚本, 不加载策略")
    ap.add_argument("--real", action="store_true",
                    help="连接真机电机: 普通sim2sim中输出仿真目标角; 与 --manual 组合时为滑块联调")
    ap.add_argument("--manual", action="store_true",
                    help="打开关节角拖动滑块面板 (可与 --real 组合)")
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
