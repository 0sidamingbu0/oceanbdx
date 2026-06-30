/*
 * OceanBDX - robot configuration loaded from YAML
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_CONFIG_HPP
#define OCEANBDX_CONFIG_HPP

#include <string>
#include <vector>

namespace oceanbdx
{

struct LegPortConfig
{
    std::string port;                 // /dev/ttyleft 或 /dev/ttyright
    std::vector<int> motor_ids;       // 485总线上的电机ID (从1开始)
    std::vector<int> joint_indices;   // 对应关节向量中的下标
};

struct Config
{
    // ---- 关节 ----
    int num_joints = 10;
    std::vector<std::string> joint_names;

    // ---- 硬件 ----
    LegPortConfig left_leg;
    LegPortConfig right_leg;
    double gear_ratio = 6.33;          // GO-M8010-6 减速比

    std::string imu_port = "/dev/ttyimu";
    int imu_baud = 460800;

    std::string neck_port = "/dev/ttyneck";
    int neck_baud = 1000000;
    bool neck_enabled = false;         // 脖子暂不启用, 仅移植驱动

    std::string gamepad_device = "/dev/input/js0";  // USB 手柄 (Linux joystick)
    std::string battery_port = "/dev/ttybat";       // 电池 BMS 串口
    int battery_baud = 9600;
    bool battery_enabled = false;      // 电池监控默认关闭, 按需启用

    // ---- 标定 (URDF零位 = 站立姿态) ----
    // q_urdf = direction[i] * (q_motor - q_motor_offset[i]) + urdf_offset[i]
    std::vector<double> directions;            // 电机正方向与URDF正方向关系 (+1/-1)
    std::vector<double> q_motor_offset;        // 电机在限位位置的角度 (rad, 实测)
    std::vector<double> urdf_offset;           // 限位位置在URDF坐标系下的角度 (rad, 由URDF测量)
    std::vector<double> sit_pose;              // 坐姿启动时各关节的URDF角度 (rad)
    std::vector<double> stand_pose;            // 站立目标角度 (URDF零位附近, 通常全0)
    double boot_tolerance = 0.30;              // 上电标定时允许的姿态误差 (rad)

    // ---- 控制 ----
    double control_dt = 0.005;                 // 主控制周期 (s)
    int decimation = 4;                        // 策略周期 = control_dt * decimation
    std::vector<double> fixed_kp;              // 脚本控制(坐/起立)用增益
    std::vector<double> fixed_kd;
    std::vector<double> rl_kp;                 // RL模式增益
    std::vector<double> rl_kd;
    std::vector<double> torque_limits;
    std::vector<double> joint_lower;           // 关节软限位 (URDF坐标)
    std::vector<double> joint_upper;
    double sit_align_duration = 3.0;           // 上电回蹲姿脚本时长 (s)
    double stand_duration = 3.0;               // 起立脚本时长 (s)
    double rl_warmup_duration = 2.0;           // RL接管时动作逐渐放大时长 (s)
    double rl_target_rate_limit = 2.0;         // RL目标角最大变化率 (rad/s, <=0关闭)
    double damping_kd = 2.0;                   // 阻尼保护模式 kd

    // ---- 策略 ----
    std::string policy_path;                   // ONNX 模型路径
    int num_obs = 0;                           // 0 = 使用ONNX输入维度
    double ang_vel_scale = 0.25;
    double dof_pos_scale = 1.0;
    double dof_vel_scale = 0.05;
    double action_scale = 0.25;
    double clip_actions = 100.0;
    double clip_obs = 100.0;
    std::vector<double> commands_scale = {2.0, 2.0, 1.0};
    double gait_cycle_period = 0.6;            // 低速行走v2 gait clock周期(s)
    std::vector<double> default_dof_pos;       // 策略动作叠加的默认关节角

    // ---- 速度指令限幅 ----
    double max_vx = 0.5;
    double max_vy = 0.3;
    double max_wz = 0.8;

    static Config Load(const std::string &yaml_path);
};

} // namespace oceanbdx

#endif // OCEANBDX_CONFIG_HPP
