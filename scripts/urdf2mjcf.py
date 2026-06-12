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
    floor.friction = [1.0, 0.005, 0.0001]

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
    os.makedirs(os.path.dirname(XML_OUT), exist_ok=True)
    open(XML_OUT, "w").write(xml)

    # 验证最终XML可加载
    model = mujoco.MjModel.from_xml_path(XML_OUT)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print(f"saved {XML_OUT}")
    print(f"nq={model.nq} nv={model.nv} joints={names}")


if __name__ == "__main__":
    main()
