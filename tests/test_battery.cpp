/*
 * 调试步骤7: 电池/BMS 测试 (A5 串口协议, /dev/ttybat)
 *
 * 用法: ./test_battery [/dev/ttybat] [9600]
 * 验证要点:
 *   - VALID 出现, 累计/采集电压在合理范围 (例如 20~30V 视电池组而定)
 *   - SOC 在 0~100%
 *   - 上电/负载变化时电流随之变化
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/battery_driver.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    std::string port = (argc > 1) ? argv[1] : "/dev/ttybat";
    int baud = (argc > 2) ? std::atoi(argv[2]) : 9600;

    signal(SIGINT, OnSig);
    oceanbdx::BatteryDriver bat(port, baud);
    if (!bat.Start()) return -1;

    while (g_run)
    {
        auto s = bat.GetState();
        printf("\033[2J\033[H=== battery %s | %s | %.1f Hz ===\n", port.c_str(),
               s.valid ? "VALID" : "WAITING", bat.UpdateHz());
        printf("cumulative_voltage: %7.2f V\n", s.cumulative_voltage);
        printf("gather_voltage:     %7.2f V\n", s.gather_voltage);
        printf("current:            %7.2f A\n", s.current);
        printf("soc:                %7.2f %%\n", s.soc);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return 0;
}
