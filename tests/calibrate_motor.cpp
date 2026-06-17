/*
 * GO-M8010-6 电机编码器零位校准工具
 *
 * 用法:
 *   ./calibrate_motor <port> <motor_id>
 *
 *   port      : 串口设备路径, 例如 /dev/ttyright 或 /dev/ttyleft
 *   motor_id  : 总线上的电机 ID (1-14)
 *
 * 执行效果:
 *   向指定电机发送一次 CALIBRATE 指令 (MotorMode::CALIBRATE, status=2),
 *   电机固件将当前转子位置记为新零点并保存到内部 Flash。
 *   操作前请先将关节手动摆到结构限位(或任意希望设为零位的位置)。
 *
 * 安全注意:
 *   - 校准会覆盖电机内部零位, 必须在机器人断电下电(不使能)状态进行
 *   - 校准后务必重新测量并更新 config/oceanbdx.yaml 中的 limit_pose
 *   - 每次运行只校准一个电机, 执行完自动退出
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

#include <cstdlib>
#include <iostream>


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "用法: " << argv[0] << " <port> <motor_id>\n"
                  << "  示例: ./calibrate_motor /dev/ttyright 1\n";
        return 1;
    }

    const std::string port_name = argv[1];
    const int motor_id = std::atoi(argv[2]);

    if (motor_id < 1 || motor_id > 14)
    {
        std::cerr << "[错误] motor_id 范围 1-14, 当前输入: " << motor_id << "\n";
        return 1;
    }

    // ---------- 打开串口 ----------
    SerialPort serial(port_name);   // 失败时抛出异常并终止
    std::cout << "串口 " << port_name << " 已打开\n"
              << "目标电机 ID: " << motor_id << "\n"
              << "发送 CALIBRATE 指令...\n";

    // ---------- 构造校准命令 ----------
    MotorCmd cmd;
    MotorData data;
    cmd.motorType  = MotorType::GO_M8010_6;
    data.motorType = MotorType::GO_M8010_6;
    cmd.id   = static_cast<unsigned short>(motor_id);
    cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::CALIBRATE);
    // 校准帧无需力矩/速度/位置目标, 全清零
    cmd.q   = 0.0f;
    cmd.dq  = 0.0f;
    cmd.kp  = 0.0f;
    cmd.kd  = 0.0f;
    cmd.tau = 0.0f;

    // ---------- 发送校准帧 (单帧) ----------
    const bool ack_received = serial.sendRecv(&cmd, &data);
    if (ack_received)
    {
        std::cout << "电机应答: q=" << data.q << " rad (转子侧)  "
                  << "mode=" << static_cast<int>(data.mode) << "  "
                  << "err=" << static_cast<int>(data.merror) << "\n";
    }
    else
    {
        std::cout << "未收到应答\n";
    }

    // ---------- 结果报告 ----------
    if (ack_received)
    {
        std::cout << "\n[完成] 电机 ID=" << motor_id << " 编码器零位已写入。\n"
                  << "  下一步: 将关节摆到坐姿/限位, 运行 test_calibration 重新标定 limit_pose\n"
                  << "          并更新 config/oceanbdx.yaml。\n";
    }
    else
    {
        std::cerr << "\n[警告] 未收到应答, 请检查:\n"
                  << "  1. 串口路径是否正确 (" << port_name << ")\n"
                  << "  2. 电机 ID 是否匹配 (当前 ID=" << motor_id << ")\n"
                  << "  3. 电机是否上电且 485 总线接线正常\n"
                  << "  4. 是否有其他进程 (oceanbdx_run/oceanbdx_teleop) 占用串口\n";
        return 1;
    }

    return 0;
}
