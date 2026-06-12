/*
 * 调试步骤5: 手柄测试 (Retroid UDP 手柄, 来自 DeepRobotics gamepad 库)
 *
 * 手机装 controlapp 连同一局域网, UDP端口默认 12121。
 * 用法: ./test_gamepad [port]
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "retroid_gamepad.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    int port = (argc > 1) ? std::atoi(argv[1]) : 12121;
    signal(SIGINT, OnSig);

    RetroidGamepad pad(port);
    pad.StartDataThread();

    while (g_run)
    {
        RetroidKeys k = pad.GetKeys();
        printf("\033[2J\033[H=== retroid gamepad udp:%d ===\n", port);
        printf("LX=%6.3f LY=%6.3f RX=%6.3f RY=%6.3f\n",
               k.left_axis_x, k.left_axis_y, k.right_axis_x, k.right_axis_y);
        printf("A=%d B=%d X=%d Y=%d L1=%d R1=%d L2=%d R2=%d start=%d select=%d\n",
               k.A, k.B, k.X, k.Y, k.L1, k.R1, k.L2, k.R2, k.start, k.select);
        printf("dpad: up=%d down=%d left=%d right=%d\n", k.up, k.down, k.left, k.right);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
