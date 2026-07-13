#!/usr/bin/env python3
"""Headless MuJoCo diagnostics for a walking policy.

Runs the normal SIT -> STAND_UP -> RL_STAND -> RL_WALK state sequence, then
reports per-foot clearance/contact, leg torque and neck motion for a fixed
velocity command. This mirrors scripts/diag_walk_neck.py on the IsaacLab side.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_sim as ms  # noqa: E402


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def box_ground_clearance(sim, geom_id):
    """Return the lowest world-space corner height for a MuJoCo box geom."""
    center_z = float(sim.data.geom_xpos[geom_id, 2])
    rotation = sim.data.geom_xmat[geom_id].reshape(3, 3)
    half_size = sim.model.geom_size[geom_id]
    vertical_extent = float(np.sum(np.abs(rotation[2, :]) * half_size))
    return center_z - vertical_extent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ms.ROOT, "config/oceanbdx.yaml"))
    parser.add_argument("--policy", default=None)
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    sim_args = SimpleNamespace(
        config=args.config,
        policy=args.policy,
        stand_policy=None,
        no_policy=False,
        real=False,
        manual=False,
        no_log=True,
        sim_rl_kd=None,
        sim_rl_kd_list=None,
        sim_action_scale=None,
    )
    sim = ms.Sim(sim_args)
    if sim.walk_policy is None:
        raise RuntimeError("walking policy is not available")

    sim_steps_per_ctrl = max(1, int(round(sim.control_dt / sim.model.opt.timestep)))
    sim.stand_start = sim.data.qpos[sim.q_adr].copy()
    sim.state, sim.state_time = "STAND_UP", 0.0
    settle_policy_steps = 0
    walk_policy_steps = 0
    walk_switch_requested = False
    walk_command_started = False
    last_rl_tick = -1
    samples = []

    while walk_policy_steps < args.steps and sim.state != "DAMPING":
        tau, neck_tau = sim.control_step()
        in_rl = sim.state in ("RL_STAND", "RL_WALK") and sim.policy is not None
        policy_tick = in_rl and sim.rl_tick != last_rl_tick and (sim.rl_tick - 1) % sim.decimation == 0
        if policy_tick:
            last_rl_tick = sim.rl_tick
            if sim.state == "RL_STAND":
                settle_policy_steps += 1
                if settle_policy_steps >= args.settle_steps and not walk_switch_requested:
                    sim._switch_rl_state("RL_WALK")
                    walk_switch_requested = True
                    last_rl_tick = -1
            else:
                walk_policy_steps += 1

        if sim.state == "RL_WALK" and not walk_command_started:
            sim.cmd[:] = [args.vx, args.vy, args.wz]
            walk_command_started = True

        q = sim.data.qpos[sim.q_adr].copy()
        dq = sim.data.qvel[sim.v_adr].copy()
        tau_motor = sim.last_kp * (sim.last_target - q) - sim.last_kd * dq
        left_force, right_force = sim.foot_contact_forces()
        right_z, left_z = sim.foot_geom_heights()
        if sim.state == "RL_WALK" and walk_policy_steps > args.warmup_steps:
            _, vel_b = sim.base_linear_velocities()
            samples.append(
                {
                    "velocity": float(sim.forward_vx_sign * vel_b[0]),
                    "right_contact": right_force > 1.0,
                    "left_contact": left_force > 1.0,
                    "right_clearance": box_ground_clearance(sim, sim.foot_geom_ids[0]),
                    "left_clearance": box_ground_clearance(sim, sim.foot_geom_ids[1]),
                    "tau": np.asarray(tau).copy(),
                    "tau_motor": np.asarray(tau_motor).copy(),
                    "neck_q": sim.data.qpos[sim.neck_q_adr].copy(),
                    "neck_target": sim.neck_target.copy(),
                }
            )

        for _ in range(sim_steps_per_ctrl):
            sim.data.qfrc_applied[:] = 0.0
            sim.data.qfrc_applied[sim.v_adr] = tau
            sim.data.qfrc_applied[sim.neck_v_adr] = neck_tau
            mujoco.mj_step(sim.model, sim.data)

    if not samples:
        raise RuntimeError(f"no walking samples collected; final state={sim.state}")

    tau = np.stack([s["tau"] for s in samples])
    tau_motor = np.stack([s["tau_motor"] for s in samples])
    neck_q = np.stack([s["neck_q"] for s in samples])
    neck_target = np.stack([s["neck_target"] for s in samples])
    print("\n========== MUJOCO WALK DIAGNOSTICS ==========")
    print(f"cmd vx/vy/wz={args.vx:+.2f}/{args.vy:+.2f}/{args.wz:+.2f} samples={len(samples)}")
    print(f"state={sim.state} velocity mean={np.mean([s['velocity'] for s in samples]):+.3f}m/s")
    for side in ("right", "left"):
        contacts = np.array([s[f"{side}_contact"] for s in samples], dtype=bool)
        clearances = [s[f"{side}_clearance"] for s in samples if not s[f"{side}_contact"]]
        low = float(np.mean(np.asarray(clearances) < 0.015)) if clearances else float("nan")
        print(
            f"[{side}] swing={1.0-contacts.mean():.3f} contact={contacts.mean():.3f} "
            f"clearance p50/p95={percentile(clearances, 50)*100:.2f}/"
            f"{percentile(clearances, 95)*100:.2f}cm <1.5cm={low:.2%}"
        )
    print(
        "[torque] |tau| p99 right={:.2f}Nm left={:.2f}Nm; clip/friction delta right={:.2%} left={:.2%}".format(
            np.percentile(np.abs(tau[:, :5]), 99),
            np.percentile(np.abs(tau[:, 5:]), 99),
            np.mean(np.abs(tau_motor[:, :5] - tau[:, :5]) > 0.25),
            np.mean(np.abs(tau_motor[:, 5:] - tau[:, 5:]) > 0.25),
        )
    )
    print(f"[neck] q RMS rad={np.sqrt(np.mean(neck_q**2, axis=0)).round(4).tolist()}")
    print(f"       target RMS rad={np.sqrt(np.mean(neck_target**2, axis=0)).round(4).tolist()}")
    print("=============================================\n")


if __name__ == "__main__":
    main()
