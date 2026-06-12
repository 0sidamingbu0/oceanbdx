/*
 * OceanBDX - Disney BDX style biped robot controller
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_TYPES_HPP
#define OCEANBDX_TYPES_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace oceanbdx
{

// 全身腿部关节数 (左腿5 + 右腿5), 关节向量顺序与 URDF / IsaacLab 训练顺序一致:
// [leg_l1, leg_l2, leg_l3, leg_l4, leg_l5, leg_r1, leg_r2, leg_r3, leg_r4, leg_r5]
constexpr int kMaxJoints = 20;

struct JointState
{
    double q = 0.0;       // 关节位置 (rad, URDF坐标系, 站立姿态为零位)
    double dq = 0.0;      // 关节速度 (rad/s)
    double tau = 0.0;     // 估计力矩 (N.m)
};

struct JointCommand
{
    double q = 0.0;       // 目标位置 (rad, URDF坐标系)
    double dq = 0.0;      // 目标速度 (rad/s)
    double kp = 0.0;
    double kd = 0.0;
    double tau = 0.0;     // 前馈力矩 (N.m)
};

struct ImuState
{
    // 四元数 (w, x, y, z)
    std::array<double, 4> quat = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> gyro = {0.0, 0.0, 0.0};   // rad/s
    std::array<double, 3> accel = {0.0, 0.0, 0.0};  // m/s^2
    bool valid = false;
};

struct RobotState
{
    ImuState imu;
    std::vector<JointState> joints;
};

struct RobotCommand
{
    std::vector<JointCommand> joints;
};

// 遥控/键盘速度指令
struct VelocityCommand
{
    std::atomic<double> vx{0.0};
    std::atomic<double> vy{0.0};
    std::atomic<double> wz{0.0};
};

} // namespace oceanbdx

#endif // OCEANBDX_TYPES_HPP
