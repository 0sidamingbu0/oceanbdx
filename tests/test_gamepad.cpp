/*
 * 调试步骤6: USB 手柄测试 (Linux joystick, /dev/input/jsX)
 *
 * 适用于 USB 2.4G 无线手柄 (XInput 模式, 如罗技 F710)。
 * 用法: ./test_gamepad [/dev/input/js0]
 * 验证要点:
 *   - connected = YES, 拨动摇杆 axes 在 -1..1 平滑变化
 *   - 按键 A/B/X/Y/LB/RB 等按下时对应位变 1
 *   - 方向键体现在 axes[6]/axes[7]
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/gamepad_driver.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    std::string device = (argc > 1) ? argv[1] : "/dev/input/js0";
    signal(SIGINT, OnSig);

    oceanbdx::GamepadDriver pad(device);
    if (!pad.Start()) return -1;

    using namespace oceanbdx;
    while (g_run)
    {
        GamepadState s = pad.GetState();
        printf("\033[2J\033[H=== USB gamepad %s | %s ===\n", device.c_str(),
               s.connected ? "CONNECTED" : "DISCONNECTED");

        // --- 已知映射 ---
        printf("LX=%6.3f LY=%6.3f  RX=%6.3f RY=%6.3f  DPad(%+.0f,%+.0f)\n",
               s.axes[kAxisLeftX], s.axes[kAxisLeftY], s.axes[kAxisRightX], s.axes[kAxisRightY],
               s.axes[kAxisDpadX], s.axes[kAxisDpadY]);
        printf("A=%d B=%d X=%d Y=%d  L1=%d R1=%d  L2=%d R2=%d  select=%d start=%d  LS=%d RS=%d\n",
               s.buttons[kBtnA], s.buttons[kBtnB], s.buttons[kBtnX], s.buttons[kBtnY],
               s.buttons[kBtnL1], s.buttons[kBtnR1],
               s.buttons[kBtnL2], s.buttons[kBtnR2],
               s.buttons[kBtnSelect], s.buttons[kBtnStart],
               s.buttons[kBtnLStick], s.buttons[kBtnRStick]);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
