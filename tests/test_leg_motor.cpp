/*
 * 调试步骤1: 单腿/单电机测试 (只读模式 + 可选小增益位置保持)
 *
 * 用法:
 *   ./test_leg_motor /dev/ttyleft 5            # 只读: 轮询ID1-5并打印 (零增益, 安全)
 *   ./test_leg_motor /dev/ttyleft 5 hold       # 位置保持: 用小增益锁定当前位置
 *
 * 验证要点:
 *   - 所有电机在线, 轮询频率 > 200Hz
 *   - 手动转动关节, q读数方向/幅值正确 (输出轴弧度)
 *   - hold模式下关节有弹性阻力且不抖动
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <port> <num_motors> [hold]" << std::endl;
        return -1;
    }
    std::string port = argv[1];
    int num = std::atoi(argv[2]);
    bool hold = (argc > 3 && std::string(argv[3]) == "hold");
    const double gr = queryGearRatio(MotorType::GO_M8010_6);

    signal(SIGINT, OnSig);
    SerialPort serial(port);

    MotorCmd cmd;
    MotorData data;
    cmd.motorType = MotorType::GO_M8010_6;
    data.motorType = MotorType::GO_M8010_6;

    std::vector<double> hold_q(num, 0.0);
    std::vector<bool> hold_init(num, false);
    std::vector<double> q(num, 0), dq(num, 0), tau(num, 0);
    std::vector<bool> online(num, false);

    int loops = 0;
    auto t0 = std::chrono::steady_clock::now();
    double hz = 0;

    while (g_run)
    {
        for (int i = 0; i < num; ++i)
        {
            cmd.id = i + 1;
            cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
            cmd.q = 0; cmd.dq = 0; cmd.kp = 0; cmd.kd = 0; cmd.tau = 0;

            if (hold && hold_init[i])
            {
                cmd.q = hold_q[i] * gr;           // 转子侧
                cmd.kp = 10.0 / (gr * gr);        // 输出轴 kp=10 -> 转子侧
                cmd.kd = 0.5 / (gr * gr);
            }

            if (serial.sendRecv(&cmd, &data))
            {
                online[i] = true;
                q[i] = data.q / gr;               // 输出轴
                dq[i] = data.dq / gr;
                tau[i] = data.tau * gr;
                if (hold && !hold_init[i])
                {
                    hold_q[i] = q[i];
                    hold_init[i] = true;
                }
            }
            else
            {
                online[i] = false;
            }
        }

        if (++loops % 100 == 0)
        {
            auto t1 = std::chrono::steady_clock::now();
            hz = 100.0 * 1e6 / std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            t0 = t1;
            printf("\033[2J\033[H=== %s | %.1f Hz | mode=%s ===\n", port.c_str(), hz, hold ? "HOLD" : "READ");
            for (int i = 0; i < num; ++i)
                printf("ID %d [%s] q=%8.4f rad (%7.2f deg)  dq=%8.4f  tau=%8.4f\n",
                       i + 1, online[i] ? "OK" : "--", q[i], q[i] * 180.0 / M_PI, dq[i], tau[i]);
        }
    }

    // 退出: 发零增益
    for (int i = 0; i < num; ++i)
    {
        cmd.id = i + 1;
        cmd.q = 0; cmd.dq = 0; cmd.kp = 0; cmd.kd = 0; cmd.tau = 0;
        serial.sendRecv(&cmd, &data);
    }
    return 0;
}
