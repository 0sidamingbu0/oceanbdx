/*
 * 调试步骤4: 零位/标定测试
 *
 * 两种模式:
 *
 * [1] 监视模式 (默认):
 *     ./test_calibration config/oceanbdx.yaml
 *   实时显示 电机原始角 / 换算后URDF角 / sit_pose偏差, 用于:
 *       - 验证 directions: 正向转动关节, URDF角变化方向应与URDF可视化一致 (否则改 direction)
 *       - 验证标定: 手动摆出坐姿, URDF角应落在 sit_pose ± tolerance
 *
 * [2] 限位标定模式 (测量 q_motor_offset):
 *     ./test_calibration config/oceanbdx.yaml limit             # 默认每关节顶"下限端"
 *     ./test_calibration config/oceanbdx.yaml limit LULLLLULLL  # 逐关节指定限位侧 (L=下限/U=上限)
 *   操作:
 *       1. 先用监视模式确认 directions 已标定正确
 *       2. 把每个关节缓慢顶到指定的结构限位 (L=URDF下限端 / U=URDF上限端)
 *       3. 顶稳后按回车抓拍 -> 自动打印可直接粘贴的 q_motor_offset YAML 行
 *       4. 可反复抓拍取稳定值; 输入 q 回车退出 (或 Ctrl-C)
 *   原理: 关节顶到结构限位时读取电机输出轴角度 q_motor, 即为 q_motor_offset。
 *         urdf_offset 需另行用 scripts/measure_offset.py 测量。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/calibration.hpp"
#include "oceanbdx/config.hpp"
#include "oceanbdx/leg_driver.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace oceanbdx;

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }
static std::atomic<bool> g_snap{false};

// URDF 结构限位 (rad), 顺序与 joint_names 一致, 取自 description/urdf/ocean.urdf
//   idx: 0=r1 1=r2 2=r3 3=r4 4=r5  5=l1 6=l2 7=l3 8=l4 9=l5
// ★ 若修改了 URDF 关节限位, 必须同步更新此表。
static const double kUrdfLower[10] = {-0.436, -0.750, -1.361, -0.803, -1.396,
                                      -0.349, -0.262, -0.489, -1.344, -1.134};
static const double kUrdfUpper[10] = {0.349, 0.262, 0.489, 1.344, 1.134,
                                      0.436, 0.750, 1.361, 0.803, 1.396};

// 后台读取 stdin: 空行=抓拍, 'q'=退出
static void InputThread()
{
    std::string line;
    while (g_run && std::getline(std::cin, line))
    {
        if (line == "q" || line == "Q") { g_run = false; break; }
        g_snap = true;
    }
}

int main(int argc, char **argv)
{
    std::string config_path = (argc > 1) ? argv[1] : "config/oceanbdx.yaml";
    bool limit_mode = (argc > 2 && std::string(argv[2]) == "limit");

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

    // ---------- 监视模式 ----------
    if (!limit_mode)
    {
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
        left.SetDamping(0); right.SetDamping(0);
        return 0;
    }

    // ---------- 限位标定模式 ----------
    const int n = cfg.num_joints;
    if (n > 10)
    {
        printf("limit 模式内置 URDF 限位表仅覆盖 10 个腿部关节\n");
        return -1;
    }
    // 解析每关节限位侧 (L=下限端 / U=上限端), 默认全 L (仅用于显示提示)
    std::string side((argc > 3) ? argv[3] : std::string(n, 'L'));
    if ((int)side.size() != n)
    {
        printf("限位侧字符串长度应为 %d (实际 %zu); 改用默认全 L\n", n, side.size());
        side = std::string(n, 'L');
    }
    std::vector<double> target(n);
    for (int i = 0; i < n; ++i)
    {
        bool up = (side[i] == 'U' || side[i] == 'u');
        target[i] = up ? kUrdfUpper[i] : kUrdfLower[i];
    }

    std::thread(InputThread).detach();

    std::string snapshot;  // 上次抓拍的 YAML 输出, 持续显示
    while (g_run)
    {
        printf("\033[2J\033[H=== q_motor_offset 标定 (顶到结构限位 -> 回车抓拍 / q 退出) ===\n");
        printf("%-16s %4s %10s %12s\n", "joint", "side", "q_motor", "q_motor_offset*");
        std::vector<double> cur_offset(n);
        for (int i = 0; i < n; ++i)
        {
            double qm = jmap[i].leg->GetState(jmap[i].idx).q;
            cur_offset[i] = qm;   // 限位处的电机输出轴角度即为 q_motor_offset
            printf("%-16s %4c %10.4f %12.4f\n",
                   cfg.joint_names[i].c_str(), side[i], qm, cur_offset[i]);
        }
        printf("\n(*) q_motor_offset = 限位处的电机输出轴角度 (实时显示, 顶稳后再抓拍)\n");
        printf("    urdf_offset 请用 scripts/measure_offset.py 另行测量\n");

        if (g_snap.exchange(false))
        {
            std::string line = "    q_motor_offset: [";
            char buf[64];
            for (int i = 0; i < n; ++i)
            {
                const char *sep = (i == n - 1) ? "]" : (i == 4 ? ",\n                      " : ", ");
                snprintf(buf, sizeof(buf), "%.4f%s", cur_offset[i], sep);
                line += buf;
            }
            snapshot = "已抓拍 -> 粘贴到 config/oceanbdx.yaml 的 calibration: 下:\n" + line + "\n";
        }
        if (!snapshot.empty())
            printf("\n%s", snapshot.c_str());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    left.SetDamping(0); right.SetDamping(0);
    return 0;
}
