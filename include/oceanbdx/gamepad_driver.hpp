/*
 * OceanBDX - USB 手柄驱动 (Linux joystick API, /dev/input/jsX)
 *
 * 适用于 USB 2.4G 无线手柄 / 有线手柄, 直接读取内核 joystick 设备, 无需 ROS / WiFi。
 * 独立线程读取, 原子量无锁发布。
 *
 * 实测映射 (以本项目所用手柄为准):
 *   axes[]   : 0=左摇杆X  1=左摇杆Y  2=右摇杆X  3=右摇杆Y  6=方向键X  7=方向键Y
 *   buttons[]: 0=A  1=B  3=X  4=Y  6=L1  7=R1  8=L2  9=R2  10=SELECT  11=START  13=左摇杆  14=右摇杆
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

// 轴下标 (实测映射)
enum GamepadAxis
{
    kAxisLeftX  = 0,   // 左摇杆 X   (-1=左,  +1=右)
    kAxisLeftY  = 1,   // 左摇杆 Y   (-1=上,  +1=下)
    kAxisRightX = 2,   // 右摇杆 X   (-1=左,  +1=右)
    kAxisRightY = 3,   // 右摇杆 Y   (-1=上,  +1=下)
    kAxisDpadX  = 6,   // 方向键 X   (-1=左,  +1=右)
    kAxisDpadY  = 7,   // 方向键 Y   (-1=上,  +1=下)
};

// 按键下标 (实测映射)
// 注: button[2]/button[5] 此手柄无对应物理按键
enum GamepadButton
{
    kBtnA      = 0,   // A
    kBtnB      = 1,   // B
    kBtnX      = 3,   // X
    kBtnY      = 4,   // Y
    kBtnL1     = 6,   // L1
    kBtnR1     = 7,   // R1
    kBtnL2     = 8,   // L2 扳机
    kBtnR2     = 9,   // R2 扳机
    kBtnSelect = 10,  // SELECT
    kBtnStart  = 11,  // START
    kBtnLStick = 13,  // 左摇杆按下
    kBtnRStick = 14,  // 右摇杆按下
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
