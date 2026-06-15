/*
 * OceanBDX - USB 手柄驱动 (Linux joystick API, /dev/input/jsX)
 *
 * 适用于 USB 2.4G 无线手柄 / 有线手柄 (XInput 模式, 如罗技 F710),
 * 直接读取内核 joystick 设备, 无需 ROS / WiFi。独立线程读取, 原子量无锁发布。
 *
 * 标准映射 (与 ros2 joy_node 一致):
 *   axes[]   : 0=左摇杆X 1=左摇杆Y 2=LT 3=右摇杆X 4=右摇杆Y 5=RT 6=方向键X 7=方向键Y
 *   buttons[]: 0=A 1=B 2=X 3=Y 4=LB 5=RB 6=back 7=start 8=power 9=左摇杆 10=右摇杆
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_GAMEPAD_DRIVER_HPP
#define OCEANBDX_GAMEPAD_DRIVER_HPP

#include "oceanbdx/types.hpp"

#include <atomic>
#include <string>
#include <thread>

namespace oceanbdx
{

// 轴下标 (Linux joystick 标准映射)
enum GamepadAxis
{
    kAxisLeftX = 0,
    kAxisLeftY = 1,
    kAxisLT = 2,
    kAxisRightX = 3,
    kAxisRightY = 4,
    kAxisRT = 5,
    kAxisDpadX = 6,
    kAxisDpadY = 7,
};

// 按键下标 (Linux joystick 标准映射)
enum GamepadButton
{
    kBtnA = 0,
    kBtnB = 1,
    kBtnX = 2,
    kBtnY = 3,
    kBtnLB = 4,
    kBtnRB = 5,
    kBtnBack = 6,
    kBtnStart = 7,
    kBtnPower = 8,
    kBtnLStick = 9,
    kBtnRStick = 10,
};

class GamepadDriver
{
public:
    explicit GamepadDriver(const std::string &device = "/dev/input/js0");
    ~GamepadDriver();

    bool Start();
    void Stop();

    // 无锁读取当前状态快照
    GamepadState GetState() const;
    bool IsConnected() const { return connected_.load(); }

private:
    void ReadLoop();

    std::string device_;
    int fd_ = -1;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};

    // 双缓冲无锁发布
    GamepadState buf_[2];
    std::atomic<int> front_{0};
};

} // namespace oceanbdx

#endif // OCEANBDX_GAMEPAD_DRIVER_HPP
