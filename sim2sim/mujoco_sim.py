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
    python3 sim2sim/mujoco_sim.py --real          # 真机镜像: 读编码器→sim可视化 (可手动搬动)
    python3 sim2sim/mujoco_sim.py --real --manual # 真机联调: 拖动滑块→PD下发到真机
    # ★ --real 需先安装 unitree_actuator_sdk, 且停掉 C++ 主控 oceanbdx_run (串口互斥)

键盘 (MuJoCo viewer 窗口内):
    1 = 起立   2 = 行走   3 = 回平衡   9 = 阻尼   r = 重置
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

    def reset(self):
        self.last_actions[:] = 0

    def step(self, q, dq, quat, gyro, cmd):
        c = self.cfg
        obs = np.concatenate([
            gyro * c["ang_vel_scale"],
            quat_rotate_inverse_gravity(quat),
            np.asarray(cmd) * self.commands_scale,
            (q - self.default_dof_pos) * c["dof_pos_scale"],
            dq * c["dof_vel_scale"],
            self.last_actions,
        ]).astype(np.float32)
        obs = np.clip(obs, -c["clip_obs"], c["clip_obs"])
        act = self.sess.run(None, {self.input_name: obs[None, :]})[0][0]
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

        self.policy = None
        if not args.no_policy:
            pol_path = args.policy or os.path.join(ROOT, full["policy"]["path"])
            if os.path.exists(pol_path):
                pcfg = dict(full["policy"])
                pcfg["default_dof_pos"] = pcfg["default_dof_pos"][:self.nj]
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

        # 虚拟坐凳 (复刻真机底座): 坐姿时支撑躯干, 起立完成后下沉移除
        self.stool_gid = self.model.geom("stool").id
        self.stool_pos0 = self.model.geom_pos[self.stool_gid].copy()
        # 坐姿底座高度 = 凳子顶面 + base底部到原点距离 (0.1853, 由mesh测得)
        stool_top = self.stool_pos0[2] + self.model.geom_size[self.stool_gid][2]
        self.sit_base_height = stool_top + 0.1853 + 0.001  # 1mm 余量

        self.reset()

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

    def key_cb(self, keycode):
        c = chr(keycode) if keycode < 256 else ""
        if c == "1" and self.state == "SIT":
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
            print("[FSM] -> DAMPING")
        elif c in ("r", "R"):
            self.reset()
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
        if c in "123":
            return
        print(f"cmd = {self.cmd}")

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
        if self.want_real or self.want_manual:
            return self._run_teleop()
        sim_steps_per_ctrl = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        print(f"[sim2sim] sim_dt={self.model.opt.timestep} control_dt={self.control_dt} "
              f"({sim_steps_per_ctrl} substeps), policy {'ON' if self.policy else 'OFF'}")
        print(__doc__)

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_cb) as v:
            while v.is_running():
                t0 = time.time()
                tau, neck_tau = self.control_step()
                for _ in range(sim_steps_per_ctrl):
                    self.data.qfrc_applied[self.v_adr] = tau
                    self.data.qfrc_applied[self.neck_v_adr] = neck_tau
                    mujoco.mj_step(self.model, self.data)
                v.sync()
                dt_left = self.control_dt - (time.time() - t0)
                if dt_left > 0:
                    time.sleep(dt_left)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config/oceanbdx.yaml"))
    ap.add_argument("--policy", default=None, help="覆盖config中的policy路径")
    ap.add_argument("--no-policy", action="store_true", help="仅验证起立脚本, 不加载策略")
    ap.add_argument("--real", action="store_true",
                    help="连接真机电机: sim 作数字孪生镜像真机状态 (需先停掉 oceanbdx_run)")
    ap.add_argument("--manual", action="store_true",
                    help="打开关节角拖动滑块面板 (可与 --real 组合)")
    args = ap.parse_args()

    if not os.path.exists(SCENE_XML):
        print("scene xml missing, run: python3 scripts/urdf2mjcf.py")
        sys.exit(1)
    Sim(args).run()
