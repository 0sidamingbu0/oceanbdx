#!/usr/bin/env python3
"""
OceanBDX 零位偏移测量工具 (URDF可视化)

用途: 测量 "结构限位位置" 与 "站立位置(URDF零位)" 的差值, 填入
config/oceanbdx.yaml 的 calibration.limit_pose。

原理:
  电机零位在装配时设置于结构限位处, 因此
      q_urdf = direction * q_motor + limit_pose
  其中 limit_pose 即限位位置在URDF坐标系下的角度。

用法 (在 oceanbdx 根目录):
    python3 scripts/measure_offset.py

操作:
  - MuJoCo viewer 会以可拖动滑块的方式显示模型 (双击关节/用左侧joint面板拖动)
  - 把每个关节拖到结构限位位置 (与实物对照!)
  - 终端每秒打印当前所有关节角度, 即可读出 limit_pose
  - 站立姿态 = 全部归零 (URDF零位)
  - 同样方法可把模型摆成坐姿, 读出 sit_pose

注意: URDF的limit上下限不一定就是结构限位, 以实物为准。
"""
import os
import time

import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(ROOT, "sim2sim/ocean_scene.xml")

JOINTS = ["leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
          "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
          "neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint"]


def main():
    if not os.path.exists(SCENE_XML):
        print("scene xml missing, run: python3 scripts/urdf2mjcf.py")
        return
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 固定基座在空中, 关闭重力, 便于摆姿势
    model.opt.gravity[:] = 0
    data.qpos[2] = 0.8
    mujoco.mj_forward(model, data)

    q_adr = {j: model.joint(j).qposadr[0] for j in JOINTS}
    lim = {j: (model.joint(j).range[0], model.joint(j).range[1]) for j in JOINTS}

    print(__doc__)
    with mujoco.viewer.launch_passive(model, data) as v:
        last = 0.0
        while v.is_running():
            mujoco.mj_forward(model, data)
            v.sync()
            now = time.time()
            if now - last > 1.0:
                last = now
                print("\n--- 当前关节角度 (rad) | [URDF下限, 上限] ---")
                for j in JOINTS:
                    q = data.qpos[q_adr[j]]
                    print(f"{j:16s} {q:8.4f}   [{lim[j][0]:7.3f}, {lim[j][1]:7.3f}]")
                print("limit_pose 行 (按当前姿势, 腿部10关节):")
                vals = ", ".join(f"{data.qpos[q_adr[j]]:.3f}" for j in JOINTS[:10])
                print(f"  [{vals}]")
            time.sleep(0.02)


if __name__ == "__main__":
    main()
