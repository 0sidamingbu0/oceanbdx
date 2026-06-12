/*
 * OceanBDX - joint zero-offset / calibration management
 *
 * 电机零位与URDF零位的换算:
 *   URDF零位 = 站立姿态
 *   电机零位 = 结构限位位置 (装配时把关节顶到限位再标定电机零点)
 *   limit_pose[i] = 限位位置在URDF坐标系下的角度 (用URDF可视化工具测量)
 *
 *   q_urdf  = direction[i] * q_motor + limit_pose[i]
 *   q_motor = direction[i] * (q_urdf - limit_pose[i])    (direction为±1)
 *
 * GO-M8010-6 是转子侧单圈绝对值编码器 + 6.33减速机, 输出轴绝对位置存在多圈歧义。
 * 因此机器人必须以"坐姿"上电 (放在底座上, 各关节处于已知小角度范围内),
 * 上电后用 sit_pose 校验读数是否落在 boot_tolerance 范围内, 校验失败禁止使能。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_CALIBRATION_HPP
#define OCEANBDX_CALIBRATION_HPP

#include "oceanbdx/config.hpp"
#include <cmath>
#include <vector>

namespace oceanbdx
{

class JointCalibration
{
public:
    explicit JointCalibration(const Config &cfg)
        : directions_(cfg.directions), limit_pose_(cfg.limit_pose),
          sit_pose_(cfg.sit_pose), tolerance_(cfg.boot_tolerance) {}

    // 电机输出轴角度 -> URDF关节角度
    double MotorToUrdf(int i, double q_motor) const
    {
        return directions_[i] * q_motor + limit_pose_[i];
    }

    // URDF关节角度 -> 电机输出轴角度
    double UrdfToMotor(int i, double q_urdf) const
    {
        return directions_[i] * (q_urdf - limit_pose_[i]);
    }

    // 速度/力矩仅需方向变换
    double MotorToUrdfVel(int i, double dq_motor) const { return directions_[i] * dq_motor; }
    double UrdfToMotorVel(int i, double dq_urdf) const { return directions_[i] * dq_urdf; }
    double MotorToUrdfTau(int i, double tau_motor) const { return directions_[i] * tau_motor; }
    double UrdfToMotorTau(int i, double tau_urdf) const { return directions_[i] * tau_urdf; }

    // 上电坐姿校验: 全部关节 |q_urdf - sit_pose| < tolerance 才允许使能
    bool ValidateBootPose(const std::vector<double> &q_urdf, std::vector<int> *bad_joints = nullptr) const
    {
        bool ok = true;
        for (size_t i = 0; i < q_urdf.size() && i < sit_pose_.size(); ++i)
        {
            if (std::fabs(q_urdf[i] - sit_pose_[i]) > tolerance_)
            {
                ok = false;
                if (bad_joints) bad_joints->push_back(static_cast<int>(i));
            }
        }
        return ok;
    }

private:
    std::vector<double> directions_;
    std::vector<double> limit_pose_;
    std::vector<double> sit_pose_;
    double tolerance_;
};

} // namespace oceanbdx

#endif // OCEANBDX_CALIBRATION_HPP
