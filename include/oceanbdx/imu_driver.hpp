/*
 * OceanBDX - YIS320 IMU driver (USB串口, yesense 协议)
 * 独立线程高频读取串口并解析, 状态用原子量无锁发布。
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_IMU_DRIVER_HPP
#define OCEANBDX_IMU_DRIVER_HPP

#include "oceanbdx/types.hpp"

#include <atomic>
#include <string>
#include <thread>

extern "C" {
#include "analysis_data.h" // drivers/yis_imu
}

namespace oceanbdx
{

class ImuDriver
{
public:
    ImuDriver(const std::string &port, int baud = 460800);
    ~ImuDriver();

    bool Start();
    void Stop();

    ImuState GetState() const;
    bool IsValid() const { return valid_.load(); }
    double UpdateHz() const { return update_hz_.load(); }

private:
    void ReadLoop();

    std::string port_;
    int baud_;
    int fd_ = -1;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> valid_{false};
    std::atomic<double> update_hz_{0.0};

    // 无锁发布: 双缓冲 + 序号
    struct Sample
    {
        double quat[4] = {1, 0, 0, 0};
        double gyro[3] = {0, 0, 0};
        double accel[3] = {0, 0, 0};
    };
    Sample buf_[2];
    std::atomic<int> front_{0};

    // 协议解析缓存
    unsigned char recv_buf_[2048];
    int recv_idx_ = 0;
    protocol_info_t info_{};
};

} // namespace oceanbdx

#endif // OCEANBDX_IMU_DRIVER_HPP
