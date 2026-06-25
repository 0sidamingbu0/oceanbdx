#!/usr/bin/env python3
"""临时诊断脚本: 用 numpy 复现 (归一化+MLP) 直接驱动 MuJoCo sim2sim,
扫描某次训练 run 的多个 checkpoint, 看哪一代能在 MuJoCo 里站住。

复现 rsl-rl EmpiricalNormalization: (x-mean)/(std+1e-2), 再过 elu MLP。
与导出的 policy.onnx 完全等价, 但无需 onnxruntime/torch 之外的依赖。
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_sim as MS


class CkptPolicy:
    """与 mujoco_sim.Policy 同接口, 但权重来自 .pt (归一化已并入)。"""

    def __init__(self, mean, std, eps, W, b, cfg, nj):
        self.mean, self.std, self.eps = mean, std, eps
        self.W, self.b = W, b
        self.cfg = cfg
        self.nj = nj
        self.default_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float32)
        self.commands_scale = np.array(cfg.get("commands_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        self.reset()

    def reset(self):
        self.last_actions = np.zeros(self.nj, dtype=np.float32)
        self.last_raw_actions = np.zeros(self.nj, dtype=np.float32)
        self.last_obs = None
        self.last_projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    @staticmethod
    def _elu(x):
        return np.where(x > 0, x, np.exp(x) - 1.0)

    def _mlp(self, x):
        for i in range(len(self.W) - 1):
            x = self._elu(self.W[i] @ x + self.b[i])
        return self.W[-1] @ x + self.b[-1]

    def step(self, q, dq, quat, gyro, cmd):
        c = self.cfg
        pg = MS.quat_rotate_inverse_gravity(quat)
        obs = np.concatenate([
            gyro * c["ang_vel_scale"],
            pg,
            np.asarray(cmd) * self.commands_scale,
            (q - self.default_dof_pos) * c["dof_pos_scale"],
            dq * c["dof_vel_scale"],
            self.last_actions,
        ]).astype(np.float32)
        self.last_projected_gravity = pg.astype(np.float32)
        obs = np.clip(obs, -c["clip_obs"], c["clip_obs"])
        self.last_obs = obs.copy()
        nobs = (obs - self.mean) / (self.std + self.eps)
        act = self._mlp(nobs.astype(np.float32))
        self.last_raw_actions = act.astype(np.float32)
        act = np.clip(act, -c["clip_actions"], c["clip_actions"])
        self.last_actions = act.astype(np.float32)
        return self.default_dof_pos + c["action_scale"] * act


def load_ckpt_arrays(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["actor_state_dict"]
    mean = a["obs_normalizer._mean"].numpy().reshape(-1).astype(np.float32)
    std = a["obs_normalizer._std"].numpy().reshape(-1).astype(np.float32)
    idx = sorted(int(k.split(".")[1]) for k in a if k.startswith("mlp.") and k.endswith(".weight"))
    W = [a[f"mlp.{i}.weight"].numpy().astype(np.float32) for i in idx]
    b = [a[f"mlp.{i}.bias"].numpy().astype(np.float32) for i in idx]
    return mean, std, W, b, int(ck.get("iter", -1))


def run_one(sim, steps):
    """自动起立后纯站立(零指令)跑 steps 个 policy step, 返回是否摔倒+末态。"""
    sim.reset()
    sim.stand_start = sim.data.qpos[sim.q_adr].copy()
    sim.state, sim.state_time = "STAND_UP", 0.0
    sps = max(1, int(round(sim.control_dt / sim.model.opt.timestep)))
    rl_steps = 0
    fell = False
    max_tilt = 0.0
    while rl_steps < steps:
        tau, neck_tau = sim.control_step()
        if sim.state == "DAMPING":
            fell = True
            break
        if sim.state in ("RL_BALANCE", "RL_WALK") and (sim.rl_tick - 1) % sim.decimation == 0:
            pg = MS.quat_rotate_inverse_gravity(sim.data.qpos[3:7])
            max_tilt = max(max_tilt, float(np.hypot(pg[0], pg[1])))
            rl_steps += 1
        for _ in range(sps):
            sim.data.qfrc_applied[sim.v_adr] = tau
            sim.data.qfrc_applied[sim.neck_v_adr] = neck_tau
            mj_step(sim)
    pg = MS.quat_rotate_inverse_gravity(sim.data.qpos[3:7])
    return {
        "fell": fell, "rl_steps": rl_steps, "base_z": float(sim.data.qpos[2]),
        "base_xy": (float(sim.data.qpos[0]), float(sim.data.qpos[1])),
        "tilt": (float(pg[0]), float(pg[1])), "max_tilt": max_tilt,
    }


def mj_step(sim):
    import mujoco
    mujoco.mj_step(sim.model, sim.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(MS.ROOT, "config/oceanbdx.yaml"))
    ap.add_argument("--run-dir", required=True, help="logs/rsl_rl/.../<timestamp> 目录")
    ap.add_argument("--iters", default="", help="逗号分隔的迭代号; 空=自动选取若干")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--sim-rl-kd-list", default=None)
    args = ap.parse_args()

    a = argparse.Namespace(config=args.config, policy=None, no_policy=True, real=False,
                           manual=False, sim_rl_kd=None,
                           sim_rl_kd_list=args.sim_rl_kd_list, sim_action_scale=None)
    sim = MS.Sim(a)

    pcfg = dict(__import__("yaml").safe_load(open(args.config))["oceanbdx"]["policy"])
    pcfg["default_dof_pos"] = pcfg["default_dof_pos"][:sim.nj]
    s2s = __import__("yaml").safe_load(open(args.config))["oceanbdx"].get("sim2sim", {})
    if "action_scale" in s2s:
        pcfg["action_scale"] = s2s["action_scale"]

    files = sorted(glob.glob(os.path.join(args.run_dir, "model_*.pt")),
                   key=lambda p: int(p.split("model_")[1].split(".pt")[0]))
    if args.iters:
        want = set(int(x) for x in args.iters.split(","))
        files = [f for f in files if int(f.split("model_")[1].split(".pt")[0]) in want]
    elif len(files) > 12:
        # 自动均匀抽样
        idxs = np.linspace(0, len(files) - 1, 12).round().astype(int)
        files = [files[i] for i in sorted(set(idxs))]

    print(f"[scan] kd={np.round(sim.kd,2).tolist()} action_scale={pcfg['action_scale']} steps={args.steps}")
    print(f"[scan] {len(files)} checkpoints from {args.run_dir}")
    for f in files:
        it = int(f.split("model_")[1].split(".pt")[0])
        mean, std, W, b, _ = load_ckpt_arrays(f)
        sim.policy = CkptPolicy(mean, std, 1e-2, W, b, pcfg, sim.nj)
        r = run_one(sim, args.steps)
        status = "FELL " if r["fell"] else "stand"
        print(f"  iter {it:5d}: {status} rl_steps={r['rl_steps']:3d}/{args.steps} "
              f"base_z={r['base_z']:+.3f} xy=[{r['base_xy'][0]:+.3f},{r['base_xy'][1]:+.3f}] "
              f"tilt=[{r['tilt'][0]:+.3f},{r['tilt'][1]:+.3f}] max_tilt={r['max_tilt']:.3f}")


if __name__ == "__main__":
    main()
