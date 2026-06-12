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

        cal = full["calibration"]
        self.sit_pose = np.array(cal["sit_pose"][:self.nj])
        self.stand_pose = np.array(cal["stand_pose"][:self.nj])

        cmd_cfg = full["command"]
        self.max_vel = np.array([cmd_cfg["max_vx"], cmd_cfg["max_vy"], cmd_cfg["max_wz"]])

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

    def run(self):
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
    args = ap.parse_args()

    if not os.path.exists(SCENE_XML):
        print("scene xml missing, run: python3 scripts/urdf2mjcf.py")
        sys.exit(1)
    Sim(args).run()
