/*
 * OceanBDX - 脖子飞特(Feetech)舵机驱动 (SMS_STS 总线舵机)
 *
 * 注意: 按当前规划, 脖子部分只完成驱动移植, 控制逻辑暂不实现。
 * 本类提供初始化/读状态/写位置的最小封装, 主控制程序默认不启用 (neck_enabled=false)。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_NECK_DRIVER_HPP
#define OCEANBDX_NECK_DRIVER_HPP

#include "oceanbdx/types.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

class SMS_STS; // from drivers/feetech

namespace oceanbdx
{

class NeckDriver
{
public:
    NeckDriver(const std::string &port, int baud, std::vector<uint8_t> servo_ids = {1, 2, 3});
    ~NeckDriver();

    bool Start();
    void Stop();

    // 同步读取所有舵机状态 (位置/速度/电流), 单位换算为 rad / rad/s / N.m
    bool ReadStates(std::vector<JointState> &states);

    // 同步写入位置命令 (rad), speed_rad_s 限速, 0=最大
    bool WritePositions(const std::vector<double> &q_rad,
                        const std::vector<double> &speed_rad_s);

    int NumServos() const { return static_cast<int>(ids_.size()); }

    // 飞特舵机 0-4095 对应 0-2PI
    static uint16_t RadToTick(double q) {
        double t = q / (2.0 * M_PI) * 4096.0;
        if (t < 0) t = 0;
        if (t > 4095) t = 4095;
        return static_cast<uint16_t>(t);
    }
    static double TickToRad(int tick) { return tick / 4096.0 * 2.0 * M_PI; }

private:
    std::string port_;
    int baud_;
    std::vector<uint8_t> ids_;
    std::unique_ptr<SMS_STS> servo_;
    bool started_ = false;
};

} // namespace oceanbdx

#endif // OCEANBDX_NECK_DRIVER_HPP
