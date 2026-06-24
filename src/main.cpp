/*
 * OceanBDX - 主控制程序 (Jetson Orin Nano 部署)
 *
 * 线程结构:
 *   - 左腿485轮询线程 (LegDriver)
 *   - 右腿485轮询线程 (LegDriver)
 *   - IMU串口读取线程 (ImuDriver)
 *   - 主控制循环 (本文件, control_dt 周期): 读状态 -> FSM -> 写命令
 *   - 键盘线程: 状态切换与速度指令
 *
 * 键盘:
 *   0 = BOOT坐姿校验   1 = 起立      2 = 行走     3 = 回到平衡站立
 *   9/空格 = 阻尼软停   p = PASSIVE   w/s/a/d/q/e = 速度增减  x = 速度清零
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/calibration.hpp"
#include "oceanbdx/config.hpp"
#include "oceanbdx/fsm.hpp"
#include "oceanbdx/imu_driver.hpp"
#include "oceanbdx/leg_driver.hpp"
#include "oceanbdx/policy.hpp"
#include "oceanbdx/types.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <thread>
#include <unistd.h>

using namespace oceanbdx;

static std::atomic<bool> g_running{true};
static std::atomic<int> g_event{static_cast<int>(FsmEvent::NONE)};
static std::atomic<bool> g_selfcheck{false};
static VelocityCommand g_cmd_vel;

static void SignalHandler(int) { g_running = false; }

static std::string ResolvePolicyPath(const std::string &config_path, const std::string &policy_path)
{
    namespace fs = std::filesystem;
    if (policy_path.empty()) return policy_path;

    fs::path path(policy_path);
    if (path.is_absolute() || fs::exists(path)) return policy_path;

    fs::path cfg_dir = fs::path(config_path).parent_path();
    if (cfg_dir.empty()) cfg_dir = ".";

    // Config files live under config/, while policy paths are written relative to the repo root.
    fs::path repo_relative = cfg_dir / ".." / path;
    if (fs::exists(repo_relative))
    {
        std::cout << "[Policy] resolved " << policy_path << " -> " << repo_relative.lexically_normal().string()
                  << std::endl;
        return repo_relative.lexically_normal().string();
    }

    return policy_path;
}

static void KeyboardLoop(const Config &cfg)
{
    struct termios old_t, new_t;
    tcgetattr(STDIN_FILENO, &old_t);
    new_t = old_t;
    new_t.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_t);
    int flags = fcntl(STDIN_FILENO, F_GETFL);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    while (g_running)
    {
        char c;
        if (read(STDIN_FILENO, &c, 1) == 1)
        {
            switch (c)
            {
            case '0': g_event = static_cast<int>(FsmEvent::BOOT); break;
            case '1': g_event = static_cast<int>(FsmEvent::STAND); break;
            case '2': g_event = static_cast<int>(FsmEvent::WALK); break;
            case '3': g_event = static_cast<int>(FsmEvent::BALANCE); break;
            case '9':
            case ' ': g_event = static_cast<int>(FsmEvent::DAMP); break;
            case 'p': g_event = static_cast<int>(FsmEvent::PASSIVE_REQ); break;
            case 'w': g_cmd_vel.vx = std::min(g_cmd_vel.vx.load() + 0.1, cfg.max_vx); break;
            case 's': g_cmd_vel.vx = std::max(g_cmd_vel.vx.load() - 0.1, -cfg.max_vx); break;
            case 'a': g_cmd_vel.vy = std::min(g_cmd_vel.vy.load() + 0.1, cfg.max_vy); break;
            case 'd': g_cmd_vel.vy = std::max(g_cmd_vel.vy.load() - 0.1, -cfg.max_vy); break;
            case 'q': g_cmd_vel.wz = std::min(g_cmd_vel.wz.load() + 0.1, cfg.max_wz); break;
            case 'e': g_cmd_vel.wz = std::max(g_cmd_vel.wz.load() - 0.1, -cfg.max_wz); break;
            case 'x': g_cmd_vel.vx = 0; g_cmd_vel.vy = 0; g_cmd_vel.wz = 0; break;
            case 'c': g_selfcheck = true; break;
            default: break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &old_t);
}

int main(int argc, char **argv)
{
    std::string config_path = (argc > 1) ? argv[1] : "config/oceanbdx.yaml";
    Config cfg = Config::Load(config_path);
    cfg.policy_path = ResolvePolicyPath(config_path, cfg.policy_path);
    JointCalibration calib(cfg);

    signal(SIGINT, SignalHandler);

    // ---- 驱动 ----
    LegDriver left("left", cfg.left_leg, cfg.gear_ratio);
    LegDriver right("right", cfg.right_leg, cfg.gear_ratio);
    ImuDriver imu(cfg.imu_port, cfg.imu_baud);
    if (!left.Start() || !right.Start() || !imu.Start())
    {
        std::cerr << "driver init failed, exit" << std::endl;
        return -1;
    }

    // ---- 策略 ----
    std::unique_ptr<PolicyRunner> policy;
    if (!cfg.policy_path.empty())
    {
        policy.reset(new PolicyRunner(cfg));
        if (!policy->Load())
        {
            std::cerr << "policy load failed, RL states disabled" << std::endl;
            policy.reset();
        }
        else
        {
            std::vector<double> q = cfg.default_dof_pos;
            std::vector<double> dq(cfg.num_joints, 0.0);
            std::array<double, 4> quat{1.0, 0.0, 0.0, 0.0};
            std::array<double, 3> gyro{0.0, 0.0, 0.0};
            std::array<double, 3> cmd{0.0, 0.0, 0.0};
            policy->Step(q, dq, quat, gyro, cmd);
            double act_absmax = 0.0, raw_absmax = 0.0;
            int sat_count = 0;
            for (int i = 0; i < cfg.num_joints; ++i)
            {
                double a = std::fabs(static_cast<double>(policy->LastActions()[i]));
                double raw = std::fabs(static_cast<double>(policy->LastRawActions()[i]));
                act_absmax = std::max(act_absmax, a);
                raw_absmax = std::max(raw_absmax, raw);
                if (a > 0.98) ++sat_count;
            }
            std::cout << "[Policy] zero-stand probe: act_absmax=" << act_absmax
                      << " raw_act_absmax=" << raw_absmax
                      << " saturated=" << sat_count << "/" << cfg.num_joints << std::endl;
            policy->Reset();
        }
    }

    Fsm fsm(cfg, calib, policy.get());

    std::thread kb(KeyboardLoop, std::cref(cfg));

    // 关节状态/命令缓冲 (URDF坐标)
    RobotState state;
    state.joints.resize(cfg.num_joints);

    // 关节 -> (腿, 腿内下标) 映射
    struct Map { LegDriver *leg; int idx; };
    std::vector<Map> jmap(cfg.num_joints, {nullptr, -1});
    auto map_joint = [&](const char *leg_name, LegDriver *leg, int joint_index, int leg_index) {
        if (joint_index < 0 || joint_index >= cfg.num_joints)
        {
            std::cerr << "invalid " << leg_name << " joint index " << joint_index << std::endl;
            return false;
        }
        if (leg_index < 0 || leg_index >= leg->NumMotors())
        {
            std::cerr << "invalid " << leg_name << " motor slot " << leg_index << std::endl;
            return false;
        }
        if (jmap[joint_index].leg != nullptr)
        {
            std::cerr << "duplicate joint mapping for index " << joint_index << std::endl;
            return false;
        }
        jmap[joint_index] = {leg, leg_index};
        return true;
    };
    bool mapping_ok = true;
    for (size_t k = 0; k < cfg.left_leg.joint_indices.size(); ++k)
        mapping_ok &= map_joint("left", &left, cfg.left_leg.joint_indices[k], static_cast<int>(k));
    for (size_t k = 0; k < cfg.right_leg.joint_indices.size(); ++k)
        mapping_ok &= map_joint("right", &right, cfg.right_leg.joint_indices[k], static_cast<int>(k));
    for (int i = 0; i < cfg.num_joints; ++i)
    {
        if (jmap[i].leg == nullptr || jmap[i].idx < 0)
        {
            std::cerr << "invalid joint mapping for index " << i << " ("
                      << (i < static_cast<int>(cfg.joint_names.size()) ? cfg.joint_names[i] : "?")
                      << ")" << std::endl;
            mapping_ok = false;
        }
    }
    if (!mapping_ok)
    {
        g_running = false;
        left.SetDamping(cfg.damping_kd);
        right.SetDamping(cfg.damping_kd);
        left.Stop();
        right.Stop();
        imu.Stop();
        if (kb.joinable()) kb.join();
        return -1;
    }

    std::cout << "OceanBDX controller started. Keys: 0=boot 1=stand 2=walk "
                 "3=balance 9=damp p=passive wsadqe=vel x=stop c=selfcheck" << std::endl;

    auto next_tick = std::chrono::steady_clock::now();
    const auto period = std::chrono::microseconds(static_cast<long>(cfg.control_dt * 1e6));
    int print_count = 0;

    while (g_running)
    {
        // 1. 读状态 (电机输出轴 -> URDF坐标)
        state.imu = imu.GetState();
        for (int i = 0; i < cfg.num_joints; ++i)
        {
            JointState ms = jmap[i].leg->GetState(jmap[i].idx);
            state.joints[i].q = calib.MotorToUrdf(i, ms.q);
            state.joints[i].dq = calib.MotorToUrdfVel(i, ms.dq);
            state.joints[i].tau = calib.MotorToUrdfTau(i, ms.tau);
        }

        // 2. FSM
        FsmEvent ev = static_cast<FsmEvent>(g_event.exchange(static_cast<int>(FsmEvent::NONE)));
        std::array<double, 3> cmd_vel{g_cmd_vel.vx.load(), g_cmd_vel.vy.load(), g_cmd_vel.wz.load()};
        auto cmds = fsm.Update(state, cmd_vel, ev);

        // 3. 写命令 (URDF坐标 -> 电机输出轴), 力矩限幅
        for (int i = 0; i < cfg.num_joints; ++i)
        {
            JointCommand mc;
            mc.q = calib.UrdfToMotor(i, cmds[i].q);
            mc.dq = calib.UrdfToMotorVel(i, cmds[i].dq);
            mc.kp = cmds[i].kp;
            mc.kd = cmds[i].kd;
            double tau = calib.UrdfToMotorTau(i, cmds[i].tau);
            mc.tau = std::min(std::max(tau, -cfg.torque_limits[i]), cfg.torque_limits[i]);
            jmap[i].leg->SetCommand(jmap[i].idx, mc);
        }

        // 4. 吊起自检 (按 'c' 触发一次): 静止吊平时检查 IMU 方向/gyro/关节偏差/策略量级。
        //    判据: g≈(0,0,-1), gyro≈0, q-default≈0 时, act_absmax 应很小;
        //    给机身一个倾角后, 看 g 的 x/y 分量符号与 policy 输出方向是否合理。
        if (g_selfcheck.exchange(false))
        {
            auto g = PolicyRunner::ProjectedGravity(state.imu.quat);
            double q_dev_absmax = 0.0, dq_absmax = 0.0;
            for (int i = 0; i < cfg.num_joints; ++i)
            {
                q_dev_absmax = std::max(q_dev_absmax, std::fabs(state.joints[i].q - cfg.default_dof_pos[i]));
                dq_absmax = std::max(dq_absmax, std::fabs(state.joints[i].dq));
            }
            double act_absmax = 0.0;
            double raw_act_absmax = 0.0;
            if (policy)
            {
                for (float a : policy->LastActions())
                    act_absmax = std::max(act_absmax, std::fabs(static_cast<double>(a)));
                for (float a : policy->LastRawActions())
                    raw_act_absmax = std::max(raw_act_absmax, std::fabs(static_cast<double>(a)));
            }

            std::cout << "\n==== SELF CHECK [" << FsmStateName(fsm.State()) << "] ====" << std::endl;
            std::cout << "  imu valid=" << (state.imu.valid ? "yes" : "NO")
                      << "  hz=" << static_cast<int>(imu.UpdateHz()) << std::endl;
            std::cout << "  projected_gravity=(" << g[0] << ", " << g[1] << ", " << g[2]
                      << ")   [吊平静止应≈(0,0,-1)]" << std::endl;
            std::cout << "  gyro(rad/s)=(" << state.imu.gyro[0] << ", " << state.imu.gyro[1]
                      << ", " << state.imu.gyro[2] << ")   [静止应≈0]" << std::endl;
            std::cout << "  q-default per joint:";
            for (int i = 0; i < cfg.num_joints; ++i)
                std::cout << " " << (state.joints[i].q - cfg.default_dof_pos[i]);
            std::cout << std::endl;
            std::cout << "  q_dev_absmax=" << q_dev_absmax << "  dq_absmax=" << dq_absmax
                      << "  policy_act_absmax=" << act_absmax
                      << "  raw_act_absmax=" << raw_act_absmax
                      << (policy ? "" : " (no policy loaded)") << std::endl;
            if (policy)
            {
                std::cout << "  action:";
                for (int i = 0; i < cfg.num_joints; ++i)
                    std::cout << " " << std::fixed << std::setprecision(3) << policy->LastActions()[i];
                std::cout << std::defaultfloat << std::endl;
                std::cout << "  raw_action:";
                for (int i = 0; i < cfg.num_joints; ++i)
                    std::cout << " " << std::fixed << std::setprecision(3) << policy->LastRawActions()[i];
                std::cout << std::defaultfloat << std::endl;
            }
            std::cout << "  rl_target:";
            for (int i = 0; i < cfg.num_joints; ++i)
                std::cout << " " << std::fixed << std::setprecision(3) << fsm.RlTarget()[i];
            std::cout << std::defaultfloat << std::endl;
            std::cout << "  target-q:";
            for (int i = 0; i < cfg.num_joints; ++i)
                std::cout << " " << std::fixed << std::setprecision(3)
                          << (fsm.RlTarget()[i] - state.joints[i].q);
            std::cout << std::defaultfloat << std::endl;
            std::cout << "  policy_path=" << (cfg.policy_path.empty() ? "<empty>" : cfg.policy_path)
                      << "  action_scale=" << cfg.action_scale
                      << "  rl_target_rate_limit=" << cfg.rl_target_rate_limit
                      << "  rl_warmup_duration=" << cfg.rl_warmup_duration << std::endl;
            std::cout << "============================\n" << std::endl;
        }

        // 5. 状态打印 (1Hz)
        if (++print_count >= static_cast<int>(1.0 / cfg.control_dt))
        {
            print_count = 0;
            std::cout << "[" << FsmStateName(fsm.State()) << "] imu=" << (state.imu.valid ? "ok" : "NO")
                      << " imu_hz=" << static_cast<int>(imu.UpdateHz())
                      << " left_hz=" << static_cast<int>(left.LoopHz())
                      << " right_hz=" << static_cast<int>(right.LoopHz())
                      << " cmd=(" << cmd_vel[0] << "," << cmd_vel[1] << "," << cmd_vel[2] << ")"
                      << std::endl;
        }

        next_tick += period;
        std::this_thread::sleep_until(next_tick);
    }

    std::cout << "shutting down..." << std::endl;
    left.SetDamping(cfg.damping_kd);
    right.SetDamping(cfg.damping_kd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    left.Stop();
    right.Stop();
    imu.Stop();
    if (kb.joinable()) kb.join();
    return 0;
}
