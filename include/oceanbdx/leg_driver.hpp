/*
 * OceanBDX - 单腿驱动 (一路 USB-485, 宇树 GO-M8010-6 串联总线)
 *
 * 每条腿一个独立线程轮询总线上所有电机 (sendRecv 阻塞式)。
 * 命令/状态缓存用 std::atomic<double> 无锁交换。
 * 电机SDK的 q/dq 是转子侧, 本类内部完成减速比换算, 对外全部是
 * 输出轴角度(rad), 再经 JointCalibration 换算到URDF坐标。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_LEG_DRIVER_HPP
#define OCEANBDX_LEG_DRIVER_HPP

#include "oceanbdx/types.hpp"
#include "oceanbdx/config.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class SerialPort; // from drivers/unitree_motor

namespace oceanbdx
{

class LegDriver
{
public:
    // name: "left"/"right"; cfg.motor_ids 是总线上电机ID
    LegDriver(const std::string &name, const LegPortConfig &cfg, double gear_ratio);
    ~LegDriver();

    bool Start();   // 打开串口并启动轮询线程
    void Stop();

    int NumMotors() const { return static_cast<int>(motor_ids_.size()); }
    bool IsOnline(int idx) const { return online_[idx]->load(); }
    double LoopHz() const { return loop_hz_.load(); }

    // 输出轴单位 (rad, rad/s, N.m); kp/kd 也是输出轴侧, 内部换算到转子侧
    void SetCommand(int idx, const JointCommand &cmd);
    JointState GetState(int idx) const;

    // 阻尼/失能: kp=0, kd=damping_kd, tau=0
    void SetDamping(double kd);

private:
    void PollLoop();

    std::string name_;
    std::string port_name_;
    std::vector<int> motor_ids_;
    double gear_ratio_;

    std::unique_ptr<SerialPort> serial_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<double> loop_hz_{0.0};

    struct AtomicCmd
    {
        std::atomic<double> q{0.0}, dq{0.0}, kp{0.0}, kd{0.0}, tau{0.0};
    };
    struct AtomicState
    {
        std::atomic<double> q{0.0}, dq{0.0}, tau{0.0};
    };
    std::vector<std::unique_ptr<AtomicCmd>> cmds_;
    std::vector<std::unique_ptr<AtomicState>> states_;
    std::vector<std::unique_ptr<std::atomic<bool>>> online_;
};

} // namespace oceanbdx

#endif // OCEANBDX_LEG_DRIVER_HPP
