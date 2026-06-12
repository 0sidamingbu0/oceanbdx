/*
 * OceanBDX - 脖子飞特舵机驱动实现 (仅驱动移植, 控制暂不启用)
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/neck_driver.hpp"

#include "SCServo.h" // drivers/feetech

#include <iostream>

namespace oceanbdx
{

NeckDriver::NeckDriver(const std::string &port, int baud, std::vector<uint8_t> servo_ids)
    : port_(port), baud_(baud), ids_(std::move(servo_ids))
{
    servo_.reset(new SMS_STS());
}

NeckDriver::~NeckDriver() { Stop(); }

bool NeckDriver::Start()
{
    if (!servo_->begin(baud_, port_.c_str()))
    {
        std::cerr << "[NeckDriver] open " << port_ << " failed" << std::endl;
        return false;
    }
    started_ = true;
    return true;
}

void NeckDriver::Stop()
{
    if (started_)
    {
        servo_->end();
        started_ = false;
    }
}

bool NeckDriver::ReadStates(std::vector<JointState> &states)
{
    if (!started_) return false;
    states.resize(ids_.size());

    // 同步读: 地址56(位置L)开始, 位置2B+速度2B+负载2B+电压1B+温度1B+Moving1B+保留2B+电流2B
    uint8_t rx_packet[15];
    servo_->syncReadBegin(static_cast<uint8_t>(ids_.size()), sizeof(rx_packet), 5);
    servo_->syncReadPacketTx(ids_.data(), static_cast<uint8_t>(ids_.size()),
                             SMS_STS_PRESENT_POSITION_L, sizeof(rx_packet));

    bool all_ok = true;
    for (size_t i = 0; i < ids_.size(); ++i)
    {
        if (!servo_->syncReadPacketRx(ids_[i], rx_packet))
        {
            all_ok = false;
            continue;
        }
        int16_t position = servo_->syncReadRxPacketToWrod(15); // bit15 为方向位
        int16_t speed = servo_->syncReadRxPacketToWrod(15);
        int16_t current = static_cast<int16_t>((rx_packet[14] << 8) | rx_packet[13]);

        states[i].q = TickToRad(position);
        states[i].dq = speed * 0.732 * (2.0 * M_PI / 60.0); // RPM -> rad/s
        // KT = 20 kg.cm/A, 电流单位 mA: tau(N.m) = I(A) * 20 * 9.81 / 100
        states[i].tau = (current / 1000.0) * 20.0 * 9.81 / 100.0;
    }
    servo_->syncReadEnd();
    return all_ok;
}

bool NeckDriver::WritePositions(const std::vector<double> &q_rad,
                                const std::vector<double> &speed_rad_s)
{
    if (!started_ || q_rad.size() != ids_.size()) return false;

    std::vector<int16_t> positions(ids_.size());
    std::vector<uint16_t> speeds(ids_.size());
    std::vector<uint8_t> acc(ids_.size(), 50);

    for (size_t i = 0; i < ids_.size(); ++i)
    {
        positions[i] = static_cast<int16_t>(RadToTick(q_rad[i]));
        double rpm = (i < speed_rad_s.size()) ? std::abs(speed_rad_s[i]) * 60.0 / (2.0 * M_PI) : 0.0;
        uint16_t spd = static_cast<uint16_t>(rpm / 0.732);
        speeds[i] = std::min<uint16_t>(spd, 500);
    }
    servo_->SyncWritePosEx(const_cast<uint8_t *>(ids_.data()),
                           static_cast<uint8_t>(ids_.size()),
                           positions.data(), speeds.data(), acc.data());
    return true;
}

} // namespace oceanbdx
