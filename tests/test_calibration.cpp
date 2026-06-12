/*
 * 调试步骤4: 零位/标定测试
 *
 * 用法: ./test_calibration config/oceanbdx.yaml
 *
 * 功能:
 *   - 打开双腿驱动, 实时显示: 电机原始角度 / 换算后URDF角度 / sit_pose偏差
 *   - 用于实测 directions 和 limit_pose:
 *       1. 把关节推到结构限位, 电机读数应≈0 (限位即电机零位)
 *       2. 手动摆出坐姿, 检查URDF角度是否落在 sit_pose ± tolerance
 *       3. 正向转动关节, 检查URDF角度变化方向与URDF可视化一致 (否则改direction)
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/calibration.hpp"
#include "oceanbdx/config.hpp"
#include "oceanbdx/leg_driver.hpp"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <thread>

using namespace oceanbdx;

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    std::string config_path = (argc > 1) ? argv[1] : "config/oceanbdx.yaml";
    Config cfg = Config::Load(config_path);
    JointCalibration calib(cfg);

    signal(SIGINT, OnSig);
    LegDriver left("left", cfg.left_leg, cfg.gear_ratio);
    LegDriver right("right", cfg.right_leg, cfg.gear_ratio);
    if (!left.Start() || !right.Start()) return -1;

    struct Map { LegDriver *leg; int idx; };
    std::vector<Map> jmap(cfg.num_joints, {nullptr, -1});
    for (size_t k = 0; k < cfg.left_leg.joint_indices.size(); ++k)
        jmap[cfg.left_leg.joint_indices[k]] = {&left, (int)k};
    for (size_t k = 0; k < cfg.right_leg.joint_indices.size(); ++k)
        jmap[cfg.right_leg.joint_indices[k]] = {&right, (int)k};

    while (g_run)
    {
        printf("\033[2J\033[H=== calibration check (tolerance %.2f rad) ===\n", cfg.boot_tolerance);
        printf("%-16s %10s %10s %10s %10s %6s\n",
               "joint", "q_motor", "q_urdf", "sit_pose", "err", "ok?");
        bool all_ok = true;
        for (int i = 0; i < cfg.num_joints; ++i)
        {
            double qm = jmap[i].leg->GetState(jmap[i].idx).q;
            double qu = calib.MotorToUrdf(i, qm);
            double err = qu - cfg.sit_pose[i];
            bool ok = std::fabs(err) < cfg.boot_tolerance;
            all_ok &= ok;
            printf("%-16s %10.4f %10.4f %10.4f %10.4f %6s\n",
                   cfg.joint_names[i].c_str(), qm, qu, cfg.sit_pose[i], err, ok ? "OK" : "XX");
        }
        printf("\nboot check: %s\n", all_ok ? "PASS - 可以使能" : "FAIL - 禁止使能");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
