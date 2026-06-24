#!/usr/bin/env python3
"""
OceanBDX: URDF -> MJCF 场景转换脚本

用法 (在 oceanbdx 根目录):
    python3 scripts/urdf2mjcf.py

读取 description/urdf/ocean.urdf, 生成 sim2sim/ocean_scene.xml:
  - base_link 加 freejoint (浮动基座)
  - 添加地面 / 光照 / 摩擦等仿真要素
  - 关节顺序与URDF一致: [leg_r1..r5, leg_l1..l5, neck_n1..n4]
"""
import os
import re

import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_IN = os.path.join(ROOT, "description/urdf/ocean.urdf")
URDF_TMP = os.path.join(ROOT, "description/urdf/ocean_mj.urdf")
XML_OUT = os.path.join(ROOT, "sim2sim/ocean_scene.xml")


def make_mj_urdf():
    """给URDF打补丁: mesh路径 + mujoco编译器扩展标签"""
    urdf = open(URDF_IN).read()
    urdf = urdf.replace("package://ocean_description/meshes/", "")
    if "<mujoco>" not in urdf:
        urdf = urdf.replace(
            '<robot\n  name="ocean">',
            '<robot\n  name="ocean">\n'
            "  <mujoco>\n"
            '    <compiler meshdir="../meshes" balanceinertia="true" discardvisual="false"/>\n'
            "  </mujoco>",
        )
    open(URDF_TMP, "w").write(urdf)


def main():
    make_mj_urdf()
    spec = mujoco.MjSpec.from_file(URDF_TMP)

    # 仿真选项
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # 浮动基座
    base = spec.body("base_link")
    base.pos = [0, 0, 0.55]  # 初始高度, 略高于站立高度, 防止穿地
    base.add_freejoint()

    # 地面与光照
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.friction = [2.0, 0.02, 0.001]

    # 虚拟坐凳: 真实机器人坐姿靠底座支撑, 顶面高度使坐姿(sit_pose)时双脚正好着地.
    # 位于躯干底部后侧(x∈[-0.125,-0.085]), 与双脚(x>-0.075)留出间隙;
    # 起立完成后由 mujoco_sim.py 下沉到地面以下避免绊脚.
    stool = spec.worldbody.add_geom()
    stool.name = "stool"
    stool.type = mujoco.mjtGeom.mjGEOM_BOX
    stool.size = [0.02, 0.05, 0.0512]
    stool.pos = [-0.105, 0, 0.0512]
    stool.rgba = [0.4, 0.3, 0.2, 1]

    # 脚底碰撞: 等高 box 替代 mesh (两脚 STL 底面差 7~10mm 会导致单脚支撑翻倒).
    # 两脚 box 底面同高, 摩擦与地面一致.
    foot_r = spec.body("leg_r5_link").add_geom()
    foot_r.name = "foot_r"
    foot_r.type = mujoco.mjtGeom.mjGEOM_BOX
    foot_r.pos = [0.0256, -0.0760, -0.0192]
    foot_r.size = [0.0921, 0.0040, 0.0145]
    foot_r.friction = [2.0, 0.02, 0.001]
    foot_r.rgba = [0.8, 0.2, 0.2, 0.3]

    foot_l = spec.body("leg_l5_link").add_geom()
    foot_l.name = "foot_l"
    foot_l.type = mujoco.mjtGeom.mjGEOM_BOX
    foot_l.pos = [-0.0261, -0.0760, -0.0190]
    foot_l.size = [0.0926, 0.0040, 0.0149]
    foot_l.friction = [2.0, 0.02, 0.001]
    foot_l.rgba = [0.8, 0.2, 0.2, 0.3]

    # 碰撞体仅保留: base_link, 双脚(leg_r5/leg_l5), 凳子, 地面.
    # 排除剩余自碰撞对: 坐姿双脚靠拢, 摔倒时脚可能贴近躯干,
    # 避免 mesh-mesh 深穿透引发 mj_collideTree 栈爆炸.
    for b1, b2 in [("leg_r5_link", "leg_l5_link"),
                   ("base_link", "leg_r5_link"),
                   ("base_link", "leg_l5_link")]:
        ex = spec.add_exclude()
        ex.bodyname1 = b1
        ex.bodyname2 = b2

    # 关节阻尼/电枢 (接近真实电机特性, 可按需调整)
    # 注: 新版 MjSpec 中 damping/armature 为 per-dof 3向量
    import numpy as np
    spec.default.joint.damping = np.array([0.05, 0.05, 0.05])
    spec.default.joint.armature = 0.01

    model = spec.compile()
    xml = spec.to_xml()
    # 后处理: meshdir相对于输出XML位置; URDF导入会显式写0覆盖默认值, 这里恢复关节阻尼
    xml = xml.replace('meshdir="../meshes/"', 'meshdir="../description/meshes/"')
    xml = re.sub(r'(<joint name="(?:leg|neck)_[^"]*"[^/>]*?)armature="0" damping="0 0 0"',
                 r'\1armature="0.01" damping="0.05"', xml)
    # 剥掉除 base_link 外所有 link 的碰撞 mesh geom (无 contype=0 的那条):
    # 全 mesh 碰撞会因坐姿双脚靠拢/摔倒贴身引发 mj_collideTree 深穿透栈爆炸.
    # 碰撞仅保留 base_link(坐凳支撑) + 双脚 box + 凳子 + 地面.
    xml = re.sub(r'\n\s*<geom type="mesh" rgba="[^"]*" mesh="(?!base_link")[^"]*"/>', '', xml)
    os.makedirs(os.path.dirname(XML_OUT), exist_ok=True)
    open(XML_OUT, "w").write(xml)

    # 验证最终XML可加载
    model = mujoco.MjModel.from_xml_path(XML_OUT)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print(f"saved {XML_OUT}")
    print(f"nq={model.nq} nv={model.nv} joints={names}")


if __name__ == "__main__":
    main()
