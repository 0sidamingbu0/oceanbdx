/*
 * 调试步骤3: 脖子飞特舵机测试 (驱动验证, 控制逻辑暂不实现)
 *
 * 用法: ./test_neck [/dev/ttyneck] [115200]
 * 只读模式: 同步读取ID 1-3 的位置/速度/力矩并打印
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/neck_driver.hpp"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <thread>

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    std::string port = (argc > 1) ? argv[1] : "/dev/ttyneck";
    int baud = (argc > 2) ? std::atoi(argv[2]) : 115200;

    signal(SIGINT, OnSig);
    oceanbdx::NeckDriver neck(port, baud);
    if (!neck.Start()) return -1;

    std::vector<oceanbdx::JointState> states;
    while (g_run)
    {
        bool ok = neck.ReadStates(states);
        printf("\033[2J\033[H=== %s | read %s ===\n", port.c_str(), ok ? "OK" : "PARTIAL");
        for (size_t i = 0; i < states.size(); ++i)
            printf("servo %zu: q=%8.4f rad (%7.2f deg) dq=%8.4f tau=%8.4f\n",
                   i + 1, states[i].q, states[i].q * 180.0 / M_PI, states[i].dq, states[i].tau);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return 0;
}
