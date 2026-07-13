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
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_IN = os.path.join(ROOT, "description/urdf/ocean.urdf")
URDF_TMP = os.path.join(ROOT, "description/urdf/ocean_mj.urdf")
XML_OUT = os.path.join(ROOT, "sim2sim/ocean_scene.xml")

LEG_JOINTS = [
    "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint",
    "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
]
NECK_JOINTS = ["neck_n1_joint", "neck_n2_joint", "neck_n3_joint", "neck_n4_joint"]

GROUND_FRICTION = np.array([0.5, 0.02, 0.001])
TERRAIN_GRID = 0.05
TERRAIN_Z_OFFSET = 0.0005

# Robot starts at the origin facing world -X.  This course is therefore straight ahead.
SLOPE_ANGLE_DEG = 5.0
SLOPE_RAMP_LENGTH = 1.0
SLOPE_PLATEAU_LENGTH = 1.0
SLOPE_LENGTH = 2.0 * SLOPE_RAMP_LENGTH + SLOPE_PLATEAU_LENGTH
SLOPE_WIDTH = 1.2
SLOPE_X_MAX = -1.2
SLOPE_X_MIN = SLOPE_X_MAX - SLOPE_LENGTH
SLOPE_NCOL = int(round(SLOPE_LENGTH / TERRAIN_GRID)) + 1
SLOPE_NROW = int(round(SLOPE_WIDTH / TERRAIN_GRID)) + 1
SLOPE_PEAK_HEIGHT = SLOPE_RAMP_LENGTH * np.tan(np.deg2rad(SLOPE_ANGLE_DEG))

# The rough patch is to the robot's left (world -Y).  Interior samples match the
# IsaacLab rough task: 5 cm cells, 4 mm steps and heights relative to +/-12 mm.
ROUGH_SIZE = 2.5
ROUGH_CENTER = np.array([0.0, -2.1])
ROUGH_NCOL = int(round(ROUGH_SIZE / TERRAIN_GRID)) + 1
ROUGH_NROW = ROUGH_NCOL
ROUGH_HEIGHT_STEP = 0.004
ROUGH_HEIGHT_MAX = 0.024
ROUGH_SEED = 20260713
ROUGH_BLEND_CELLS = 5


def make_slope_heights():
    """Return physical slope heights before MuJoCo normalizes the hfield asset."""
    x = np.linspace(SLOPE_X_MIN, SLOPE_X_MAX, SLOPE_NCOL)
    distance_from_near_edge = SLOPE_X_MAX - x
    height = np.minimum.reduce([
        distance_from_near_edge,
        SLOPE_LENGTH - distance_from_near_edge,
        np.full_like(x, SLOPE_RAMP_LENGTH),
    ])
    height = np.clip(height, 0.0, None) * np.tan(np.deg2rad(SLOPE_ANGLE_DEG))
    return np.repeat(height[np.newaxis, :], SLOPE_NROW, axis=0)


def make_rough_heights():
    """Return a deterministic rough patch with a flat, blended perimeter."""
    rng = np.random.default_rng(ROUGH_SEED)
    relative = rng.integers(-3, 4, size=(ROUGH_NROW, ROUGH_NCOL)) * ROUGH_HEIGHT_STEP
    height = relative + 0.012

    row = np.arange(ROUGH_NROW)[:, np.newaxis]
    col = np.arange(ROUGH_NCOL)[np.newaxis, :]
    edge_distance = np.minimum.reduce([
        np.broadcast_to(row, height.shape),
        np.broadcast_to(ROUGH_NROW - 1 - row, height.shape),
        np.broadcast_to(col, height.shape),
        np.broadcast_to(ROUGH_NCOL - 1 - col, height.shape),
    ])
    blend = np.clip(edge_distance / ROUGH_BLEND_CELLS, 0.0, 1.0)
    return height * blend


def add_test_terrains(spec):
    """Add separately reachable slope and rough test regions around the flat spawn."""
    slope_heights = make_slope_heights()
    spec.add_hfield(
        name="slope_heightfield",
        size=[SLOPE_LENGTH / 2.0, SLOPE_WIDTH / 2.0, SLOPE_PEAK_HEIGHT, 0.05],
        nrow=SLOPE_NROW,
        ncol=SLOPE_NCOL,
        userdata=slope_heights.ravel(),
    )
    slope = spec.worldbody.add_geom()
    slope.name = "slope_terrain"
    slope.type = mujoco.mjtGeom.mjGEOM_HFIELD
    slope.hfieldname = "slope_heightfield"
    slope.pos = [(SLOPE_X_MIN + SLOPE_X_MAX) / 2.0, 0.0, TERRAIN_Z_OFFSET]
    slope.friction = GROUND_FRICTION
    slope.group = 2
    slope.rgba = [0.28, 0.50, 0.28, 1.0]

    rough_heights = make_rough_heights()
    spec.add_hfield(
        name="rough_heightfield",
        size=[ROUGH_SIZE / 2.0, ROUGH_SIZE / 2.0, ROUGH_HEIGHT_MAX, 0.05],
        nrow=ROUGH_NROW,
        ncol=ROUGH_NCOL,
        userdata=rough_heights.ravel(),
    )
    rough = spec.worldbody.add_geom()
    rough.name = "rough_terrain"
    rough.type = mujoco.mjtGeom.mjGEOM_HFIELD
    rough.hfieldname = "rough_heightfield"
    rough.pos = [ROUGH_CENTER[0], ROUGH_CENTER[1], TERRAIN_Z_OFFSET]
    rough.friction = GROUND_FRICTION
    rough.group = 2
    rough.rgba = [0.45, 0.39, 0.31, 1.0]


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
    floor.friction = GROUND_FRICTION
    floor.group = 2
    floor.rgba = [0.62, 0.64, 0.66, 1.0]

    add_test_terrains(spec)

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
    foot_r.friction = GROUND_FRICTION
    foot_r.rgba = [0.8, 0.2, 0.2, 0.3]

    foot_l = spec.body("leg_l5_link").add_geom()
    foot_l.name = "foot_l"
    foot_l.type = mujoco.mjtGeom.mjGEOM_BOX
    foot_l.pos = [-0.0261, -0.0760, -0.0190]
    foot_l.size = [0.0926, 0.0040, 0.0149]
    foot_l.friction = GROUND_FRICTION
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

    # Match the committed sim2sim dynamics.  Set imported joints explicitly so
    # regenerating the XML cannot silently replace the trained leg/neck model.
    spec.default.joint.damping = np.zeros(3)
    spec.default.joint.armature = 0.0043
    for name in LEG_JOINTS:
        joint = spec.joint(name)
        joint.damping = np.zeros(3)
        joint.armature = 0.0043
    for name in NECK_JOINTS:
        joint = spec.joint(name)
        joint.damping = np.full(3, 0.05)
        joint.armature = 0.01

    model = spec.compile()
    xml = spec.to_xml()
    # 后处理: meshdir相对于输出XML位置.
    xml = xml.replace('meshdir="../meshes/"', 'meshdir="../description/meshes/"')
    # 剥掉除 base_link 外所有 link 的碰撞 mesh geom (无 contype=0 的那条):
    # 全 mesh 碰撞会因坐姿双脚靠拢/摔倒贴身引发 mj_collideTree 深穿透栈爆炸.
    # 碰撞仅保留 base_link(坐凳支撑) + 双脚 box + 凳子 + 地面.
    xml = re.sub(r'\n\s*<geom type="mesh" rgba="[^"]*" mesh="(?!base_link")[^"]*"/>', '', xml)
    os.makedirs(os.path.dirname(XML_OUT), exist_ok=True)
    open(XML_OUT, "w").write(xml)

    # 验证最终XML可加载
    model = mujoco.MjModel.from_xml_path(XML_OUT)

    for name in LEG_JOINTS:
        dof = model.joint(name).dofadr[0]
        assert np.isclose(model.dof_armature[dof], 0.0043)
        assert np.isclose(model.dof_damping[dof], 0.0)
    for name in NECK_JOINTS:
        dof = model.joint(name).dofadr[0]
        assert np.isclose(model.dof_armature[dof], 0.01)
        assert np.isclose(model.dof_damping[dof], 0.05)
    for name in ("floor", "foot_r", "foot_l", "slope_terrain", "rough_terrain"):
        np.testing.assert_allclose(model.geom(name).friction, GROUND_FRICTION)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print(f"saved {XML_OUT}")
    print(f"nq={model.nq} nv={model.nv} joints={names}")
    print(
        f"terrain: slope {SLOPE_ANGLE_DEG:.1f}deg x=[{SLOPE_X_MIN:.1f},{SLOPE_X_MAX:.1f}], "
        f"rough {ROUGH_SIZE:.1f}m square centered at {ROUGH_CENTER.tolist()}"
    )


if __name__ == "__main__":
    main()
