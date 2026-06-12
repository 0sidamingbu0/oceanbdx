/*
 * 调试步骤2: YIS320 IMU测试
 *
 * 用法: ./test_imu [/dev/ttyimu] [460800]
 * 验证要点:
 *   - 更新频率接近IMU输出频率 (建议配置500Hz以上)
 *   - 水平静止时 quat≈(1,0,0,0), gyro≈0, accel≈(0,0,9.8)
 *   - 绕各轴旋转, gyro方向与机体坐标定义一致
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/imu_driver.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

static volatile bool g_run = true;
static void OnSig(int) { g_run = false; }

int main(int argc, char **argv)
{
    std::string port = (argc > 1) ? argv[1] : "/dev/ttyimu";
    int baud = (argc > 2) ? std::atoi(argv[2]) : 460800;

    signal(SIGINT, OnSig);
    oceanbdx::ImuDriver imu(port, baud);
    if (!imu.Start()) return -1;

    while (g_run)
    {
        auto s = imu.GetState();
        printf("\033[2J\033[H=== %s | %s | %.1f Hz ===\n", port.c_str(),
               s.valid ? "VALID" : "WAITING", imu.UpdateHz());
        printf("quat (wxyz): %8.4f %8.4f %8.4f %8.4f\n", s.quat[0], s.quat[1], s.quat[2], s.quat[3]);
        printf("gyro (rad/s): %8.4f %8.4f %8.4f\n", s.gyro[0], s.gyro[1], s.gyro[2]);
        printf("accel(m/s2):  %8.4f %8.4f %8.4f\n", s.accel[0], s.accel[1], s.accel[2]);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
