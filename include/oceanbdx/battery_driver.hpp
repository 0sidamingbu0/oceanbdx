/*
 * OceanBDX - 电池/BMS 串口驱动 (A5 协议, USB转串口)
 *
 * 主机以 20Hz 主动发送查询帧, BMS 返回电压/电流/SOC。独立线程收发, 原子量发布。
 * 默认 /dev/ttybat, 9600bps, 8N1。
 *
 * 协议 (从 sarocean rl_real_ocean 提取):
 *   查询帧 (13B): A5 40 90 08 00 00 00 00 00 00 00 00 7D
 *   应答帧 (>=12B): A5 01 90 08 <累计电压H L> <采集电压H L> <电流H L> <SOC H L> ...
 *     累计电压 = raw * 0.1 V
 *     采集电压 = raw * 0.1 V
 *     电流     = (raw - 30000) * 0.1 A
 *     SOC      = raw * 0.1 %
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_BATTERY_DRIVER_HPP
#define OCEANBDX_BATTERY_DRIVER_HPP

#include "oceanbdx/types.hpp"

#include <atomic>
#include <string>
#include <thread>

namespace oceanbdx
{

class BatteryDriver
{
public:
    BatteryDriver(const std::string &port = "/dev/ttybat", int baud = 9600);
    ~BatteryDriver();

    bool Start();
    void Stop();

    BatteryState GetState() const;
    bool IsValid() const { return valid_.load(); }
    double UpdateHz() const { return update_hz_.load(); }

private:
    void CommLoop();

    std::string port_;
    int baud_;
    int fd_ = -1;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> valid_{false};
    std::atomic<double> update_hz_{0.0};

    // 双缓冲无锁发布
    BatteryState buf_[2];
    std::atomic<int> front_{0};
};

} // namespace oceanbdx

#endif // OCEANBDX_BATTERY_DRIVER_HPP
