/*
 * OceanBDX - config YAML loader
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/config.hpp"

#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <sstream>

namespace oceanbdx
{

namespace
{
template <typename T>
T Get(const YAML::Node &n, const std::string &key, const T &def)
{
    if (n[key]) return n[key].as<T>();
    return def;
}

template <typename T>
std::vector<T> GetVec(const YAML::Node &n, const std::string &key)
{
    if (n[key]) return n[key].as<std::vector<T>>();
    return {};
}

template <typename T>
void RequireSize(const std::vector<T> &v, size_t expected, const std::string &name)
{
    if (v.size() != expected)
    {
        std::ostringstream oss;
        oss << name << " expects " << expected << " values, got " << v.size();
        throw std::runtime_error(oss.str());
    }
}

void RequireLegPort(const LegPortConfig &leg, const std::string &name)
{
    if (leg.motor_ids.empty()) throw std::runtime_error(name + ".motor_ids is empty");
    if (leg.motor_ids.size() != leg.joint_indices.size())
    {
        std::ostringstream oss;
        oss << name << " motor_ids/joint_indices size mismatch: "
            << leg.motor_ids.size() << " vs " << leg.joint_indices.size();
        throw std::runtime_error(oss.str());
    }
}
} // namespace

Config Config::Load(const std::string &yaml_path)
{
    YAML::Node root = YAML::LoadFile(yaml_path);
    YAML::Node n = root["oceanbdx"];
    if (!n) throw std::runtime_error("missing 'oceanbdx' key in " + yaml_path);

    Config c;
    c.num_joints = Get<int>(n, "num_joints", 10);
    c.joint_names = GetVec<std::string>(n, "joint_names");

    YAML::Node hw = n["hardware"];
    if (hw)
    {
        YAML::Node l = hw["left_leg"], r = hw["right_leg"];
        if (l)
        {
            c.left_leg.port = Get<std::string>(l, "port", "/dev/ttyleft");
            c.left_leg.motor_ids = GetVec<int>(l, "motor_ids");
            c.left_leg.joint_indices = GetVec<int>(l, "joint_indices");
        }
        if (r)
        {
            c.right_leg.port = Get<std::string>(r, "port", "/dev/ttyright");
            c.right_leg.motor_ids = GetVec<int>(r, "motor_ids");
            c.right_leg.joint_indices = GetVec<int>(r, "joint_indices");
        }
        c.gear_ratio = Get<double>(hw, "gear_ratio", 6.33);
        c.imu_port = Get<std::string>(hw, "imu_port", "/dev/ttyimu");
        c.imu_baud = Get<int>(hw, "imu_baud", 460800);
        c.neck_port = Get<std::string>(hw, "neck_port", "/dev/ttyneck");
        c.neck_baud = Get<int>(hw, "neck_baud", 1000000);
        c.neck_enabled = Get<bool>(hw, "neck_enabled", false);
        c.gamepad_device = Get<std::string>(hw, "gamepad_device", "/dev/input/js0");
        c.battery_port = Get<std::string>(hw, "battery_port", "/dev/ttybat");
        c.battery_baud = Get<int>(hw, "battery_baud", 9600);
        c.battery_enabled = Get<bool>(hw, "battery_enabled", false);
    }

    YAML::Node cal = n["calibration"];
    if (cal)
    {
        c.directions = GetVec<double>(cal, "directions");
        c.q_motor_offset = GetVec<double>(cal, "q_motor_offset");
        c.urdf_offset = GetVec<double>(cal, "urdf_offset");
        c.sit_pose = GetVec<double>(cal, "sit_pose");
        c.stand_pose = GetVec<double>(cal, "stand_pose");
        c.boot_tolerance = Get<double>(cal, "boot_tolerance", 0.30);
    }

    YAML::Node ctrl = n["control"];
    if (ctrl)
    {
        c.control_dt = Get<double>(ctrl, "dt", 0.005);
        c.decimation = Get<int>(ctrl, "decimation", 4);
        c.fixed_kp = GetVec<double>(ctrl, "fixed_kp");
        c.fixed_kd = GetVec<double>(ctrl, "fixed_kd");
        c.rl_kp = GetVec<double>(ctrl, "rl_kp");
        c.rl_kd = GetVec<double>(ctrl, "rl_kd");
        c.torque_limits = GetVec<double>(ctrl, "torque_limits");
        c.joint_lower = GetVec<double>(ctrl, "joint_lower");
        c.joint_upper = GetVec<double>(ctrl, "joint_upper");
        c.sit_align_duration = Get<double>(ctrl, "sit_align_duration", 3.0);
        c.stand_duration = Get<double>(ctrl, "stand_duration", 3.0);
        c.rl_warmup_duration = Get<double>(ctrl, "rl_warmup_duration", 2.0);
        c.rl_target_rate_limit = Get<double>(ctrl, "rl_target_rate_limit", 2.0);
        c.damping_kd = Get<double>(ctrl, "damping_kd", 2.0);
    }

    YAML::Node pol = n["policy"];
    if (pol)
    {
        c.policy_path = Get<std::string>(pol, "path", "");
        c.num_obs = Get<int>(pol, "num_obs", 0);
        c.ang_vel_scale = Get<double>(pol, "ang_vel_scale", 0.25);
        c.dof_pos_scale = Get<double>(pol, "dof_pos_scale", 1.0);
        c.dof_vel_scale = Get<double>(pol, "dof_vel_scale", 0.05);
        c.action_scale = Get<double>(pol, "action_scale", 0.25);
        c.clip_actions = Get<double>(pol, "clip_actions", 100.0);
        c.clip_obs = Get<double>(pol, "clip_obs", 100.0);
        auto cs = GetVec<double>(pol, "commands_scale");
        if (cs.size() == 3) c.commands_scale = cs;
        c.gait_cycle_period = Get<double>(pol, "gait_cycle_period", 0.6);
        c.default_dof_pos = GetVec<double>(pol, "default_dof_pos");
    }

    YAML::Node cmd = n["command"];
    if (cmd)
    {
        c.max_vx = Get<double>(cmd, "max_vx", 0.5);
        c.max_vy = Get<double>(cmd, "max_vy", 0.3);
        c.max_wz = Get<double>(cmd, "max_wz", 0.8);
    }

    // 缺省值填充
    auto fill = [&](std::vector<double> &v, double def) {
        if (v.empty()) v.assign(c.num_joints, def);
    };
    fill(c.directions, 1.0);
    fill(c.q_motor_offset, 0.0);
    fill(c.urdf_offset, 0.0);
    fill(c.sit_pose, 0.0);
    fill(c.stand_pose, 0.0);
    fill(c.fixed_kp, 40.0);
    fill(c.fixed_kd, 2.0);
    fill(c.rl_kp, 40.0);
    fill(c.rl_kd, 1.0);
    fill(c.torque_limits, 20.0);
    fill(c.joint_lower, -3.14);
    fill(c.joint_upper, 3.14);
    fill(c.default_dof_pos, 0.0);

    const size_t nj = static_cast<size_t>(c.num_joints);
    RequireLegPort(c.left_leg, "hardware.left_leg");
    RequireLegPort(c.right_leg, "hardware.right_leg");
    RequireSize(c.joint_names, nj, "joint_names");
    RequireSize(c.directions, nj, "calibration.directions");
    RequireSize(c.q_motor_offset, nj, "calibration.q_motor_offset");
    RequireSize(c.urdf_offset, nj, "calibration.urdf_offset");
    RequireSize(c.sit_pose, nj, "calibration.sit_pose");
    RequireSize(c.stand_pose, nj, "calibration.stand_pose");
    RequireSize(c.fixed_kp, nj, "control.fixed_kp");
    RequireSize(c.fixed_kd, nj, "control.fixed_kd");
    RequireSize(c.rl_kp, nj, "control.rl_kp");
    RequireSize(c.rl_kd, nj, "control.rl_kd");
    RequireSize(c.torque_limits, nj, "control.torque_limits");
    RequireSize(c.joint_lower, nj, "control.joint_lower");
    RequireSize(c.joint_upper, nj, "control.joint_upper");
    RequireSize(c.default_dof_pos, nj, "policy.default_dof_pos");

    return c;
}

} // namespace oceanbdx
