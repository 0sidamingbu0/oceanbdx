#include <unistd.h>
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <iomanip>
#include <fcntl.h>
#include <termios.h>
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

// 使用原子变量存储电机数据，实现无锁访问
struct AtomicMotorData {
    std::atomic<float> q{0.0};      // 位置
    std::atomic<float> dq{0.0};     // 速度  
    std::atomic<float> tau{0.0};    // 力矩
};

// 电池数据结构 - 使用原子变量
struct AtomicBatteryData {
    std::atomic<float> cumulative_voltage{0.0};  // 累计总压 (V)
    std::atomic<float> gather_voltage{0.0};      // 采集总压 (V)
    std::atomic<float> current{0.0};            // 电流 (A)
    std::atomic<float> soc{0.0};                // SOC (%)
    std::atomic<bool> valid{false};             // 数据有效性
    std::atomic<long long> last_update_ms{0};   // 最后更新时间(毫秒)
};

// 全局变量 - 完全无锁设计
AtomicBatteryData atomic_battery_data;
std::atomic<bool> battery_thread_running(true);
std::atomic<bool> log_thread_running(true);
std::atomic<bool> left_motor_thread_running(true);
std::atomic<bool> right_motor_thread_running(true);

// 原子电机数据全局变量 - 无锁访问
AtomicMotorData atomic_leg_l1, atomic_leg_l2, atomic_leg_l3, atomic_leg_l4, atomic_leg_l5;
AtomicMotorData atomic_leg_r1, atomic_leg_r2, atomic_leg_r3, atomic_leg_r4, atomic_leg_r5;
AtomicMotorData atomic_neck_n1;

// 频率测量全局变量
std::atomic<float> left_motor_frequency(0.0);
std::atomic<float> right_motor_frequency(0.0);

// 日志显示线程函数 - 无锁快速读取
void logDisplayThread() {
    auto last_display_time = std::chrono::steady_clock::now();
    const auto display_interval = std::chrono::milliseconds(10); // 100Hz显示频率
    
    while (log_thread_running) {
        auto current_time = std::chrono::steady_clock::now();
        
        if (current_time - last_display_time >= display_interval) {
            // 清屏
            system("clear");
            
            printf("================== Motor Status ==================\n");
            printf("Left Motor Thread: %.1f Hz | Right Motor Thread: %.1f Hz\n", 
                   left_motor_frequency.load(), right_motor_frequency.load());
            
            // 显示电池信息（无锁原子读取）
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            auto last_update_ms = atomic_battery_data.last_update_ms.load();
            auto time_since_update = now_ms - last_update_ms;
            
            if (atomic_battery_data.valid.load() && time_since_update <= 500) {
                printf("Battery: CumVolt: %6.1fV | GathVolt: %6.1fV | Current: %6.1fA | SOC: %5.1f%%\n",
                       atomic_battery_data.cumulative_voltage.load(), 
                       atomic_battery_data.gather_voltage.load(), 
                       atomic_battery_data.current.load(), 
                       atomic_battery_data.soc.load());
            } else {
                printf("Battery: No Data Available\n");
            }
            printf("=========================================================\n\n");
            
            // 无锁读取左腿关节数据 - 瞬时读取当前缓存值
            printf("Left Leg Joint 1 (ID: 1) | q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_l1.q.load(), atomic_leg_l1.q.load() * 180.0 / M_PI, atomic_leg_l1.dq.load(), atomic_leg_l1.tau.load());
            printf("Left Leg Joint 2 (ID: 2) | q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_l2.q.load(), atomic_leg_l2.q.load() * 180.0 / M_PI, atomic_leg_l2.dq.load(), atomic_leg_l2.tau.load());
            printf("Left Leg Joint 3 (ID: 3) | q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_l3.q.load(), atomic_leg_l3.q.load() * 180.0 / M_PI, atomic_leg_l3.dq.load(), atomic_leg_l3.tau.load());
            printf("Left Leg Joint 4 (ID: 4) | q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_l4.q.load(), atomic_leg_l4.q.load() * 180.0 / M_PI, atomic_leg_l4.dq.load(), atomic_leg_l4.tau.load());
            printf("Left Leg Joint 5 (ID: 5) | q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_l5.q.load(), atomic_leg_l5.q.load() * 180.0 / M_PI, atomic_leg_l5.dq.load(), atomic_leg_l5.tau.load());
            printf("\n");
            
            // 无锁读取右腿关节数据 - 瞬时读取当前缓存值  
            printf("Right Leg Joint 1 (ID: 1)| q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_r1.q.load(), atomic_leg_r1.q.load() * 180.0 / M_PI, atomic_leg_r1.dq.load(), atomic_leg_r1.tau.load());
            printf("Right Leg Joint 2 (ID: 2)| q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_r2.q.load(), atomic_leg_r2.q.load() * 180.0 / M_PI, atomic_leg_r2.dq.load(), atomic_leg_r2.tau.load());
            printf("Right Leg Joint 3 (ID: 3)| q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_r3.q.load(), atomic_leg_r3.q.load() * 180.0 / M_PI, atomic_leg_r3.dq.load(), atomic_leg_r3.tau.load());
            printf("Right Leg Joint 4 (ID: 4)| q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_r4.q.load(), atomic_leg_r4.q.load() * 180.0 / M_PI, atomic_leg_r4.dq.load(), atomic_leg_r4.tau.load());
            printf("Right Leg Joint 5 (ID: 5)| q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_leg_r5.q.load(), atomic_leg_r5.q.load() * 180.0 / M_PI, atomic_leg_r5.dq.load(), atomic_leg_r5.tau.load());
            printf("\n");
            
            // 无锁读取脖子关节数据 - 瞬时读取当前缓存值
            printf("Neck Joint (ID: 6)        | q: %7.3f rad (%7.2f°) | W: %8.4f | tau: %8.4f\n", 
                   atomic_neck_n1.q.load(), atomic_neck_n1.q.load() * 180.0 / M_PI, atomic_neck_n1.dq.load(), atomic_neck_n1.tau.load());
            printf("\n");
            
            last_display_time = current_time;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// 电池通信线程函数
void batteryCommThread() {
    int battery_fd = -1;
    
    try {
        // 打开串口
        battery_fd = open("/dev/ttybat", O_RDWR | O_NOCTTY | O_NDELAY);
        if (battery_fd == -1) {
            std::cerr << "Failed to open /dev/ttybat" << std::endl;
            return;
        }
        
        // 配置串口参数
        struct termios options;
        tcgetattr(battery_fd, &options);
        
        // 设置波特率9600
        cfsetispeed(&options, B9600);
        cfsetospeed(&options, B9600);
        
        // 设置数据位、停止位、校验位
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;   // 无校验
        options.c_cflag &= ~CSTOPB;   // 1个停止位
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;       // 8个数据位
        
        // 设置为原始模式
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        options.c_iflag &= ~(IXON | IXOFF | IXANY);
        
        // 设置超时
        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = 1;  // 0.1秒超时
        
        // 应用配置
        tcsetattr(battery_fd, TCSANOW, &options);
        tcflush(battery_fd, TCIOFLUSH);
        
        // 发送命令数据包
        uint8_t cmd_packet[] = {0xA5, 0x40, 0x90, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D};
        uint8_t recv_buffer[32];
        
        auto last_send_time = std::chrono::steady_clock::now();
        const auto send_interval = std::chrono::milliseconds(50); // 20Hz = 50ms
        
        while (battery_thread_running) {
            auto current_time = std::chrono::steady_clock::now();
            
            // 20Hz发送频率
            if (current_time - last_send_time >= send_interval) {
                // 发送命令
                ssize_t sent = write(battery_fd, cmd_packet, sizeof(cmd_packet));
                last_send_time = current_time;
                
                if (sent == sizeof(cmd_packet)) {
                    // 尝试接收响应
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                    ssize_t recv_len = read(battery_fd, recv_buffer, sizeof(recv_buffer));
                    
                    if (recv_len >= 12) { // 至少需要12字节的响应
                        // 查找有效数据包 A5 01 90 08
                        for (ssize_t i = 0; i <= recv_len - 12; i++) {
                            if (recv_buffer[i] == 0xA5 && recv_buffer[i+1] == 0x01 && 
                                recv_buffer[i+2] == 0x90 && recv_buffer[i+3] == 0x08) {
                                
                                // 解析数据
                                uint16_t cumulative_raw = (recv_buffer[i+4] << 8) | recv_buffer[i+5];
                                uint16_t gather_raw = (recv_buffer[i+6] << 8) | recv_buffer[i+7];
                                uint16_t current_raw = (recv_buffer[i+8] << 8) | recv_buffer[i+9];
                                uint16_t soc_raw = (recv_buffer[i+10] << 8) | recv_buffer[i+11];
                                
                                // 更新全局数据（无锁原子操作）
                                atomic_battery_data.cumulative_voltage.store(cumulative_raw * 0.1f);
                                atomic_battery_data.gather_voltage.store(gather_raw * 0.1f);
                                atomic_battery_data.current.store((current_raw - 30000) * 0.1f);
                                atomic_battery_data.soc.store(soc_raw * 0.1f);
                                atomic_battery_data.valid.store(true);
                                
                                auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count();
                                atomic_battery_data.last_update_ms.store(now_ms);
                                break;
                            }
                        }
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (const std::exception& e) {
        std::cerr << "Battery communication error: " << e.what() << std::endl;
    }
    
    // 关闭串口
    if (battery_fd != -1) {
        close(battery_fd);
    }
}

// 左腿电机控制线程函数 - 最高频率运行，无锁更新
void leftMotorControlThread() {
    SerialPort serialLeft("/dev/ttyleft");
    MotorCmd cmd;
    cmd.motorType = MotorType::GO_M8010_6;
    
    // 本地电机数据变量
    MotorData leg_l1_joint, leg_l2_joint, leg_l3_joint, leg_l4_joint, leg_l5_joint, neck_n1_joint;
    MotorData* leg_l_joints[5] = {&leg_l1_joint, &leg_l2_joint, &leg_l3_joint, &leg_l4_joint, &leg_l5_joint};
    
    // 初始化电机类型
    for(int i = 0; i < 5; i++) {
        leg_l_joints[i]->motorType = MotorType::GO_M8010_6;
    }
    neck_n1_joint.motorType = MotorType::GO_M8010_6;
    
    // 频率测量变量
    auto last_freq_update = std::chrono::steady_clock::now();
    int loop_count = 0;
    
    while (left_motor_thread_running) {
        // 控制左腿5个关节 (ID 1-5) - 阻塞获取数据后立即更新原子变量
        for(int i = 1; i < 6; i++) {
            cmd.id = i;
            cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
            cmd.kp = 0.0;
            cmd.kd = 0.0;
            cmd.q = 0.0;
            cmd.dq = 0.0 * queryGearRatio(MotorType::GO_M8010_6);
            cmd.tau = 0.0;
            
            // 发送命令并获取数据（阻塞）
            serialLeft.sendRecv(&cmd, leg_l_joints[i-1]);
            
            // 立即更新原子变量（无锁）
            switch(i) {
                case 1: 
                    atomic_leg_l1.q.store(leg_l1_joint.q);
                    atomic_leg_l1.dq.store(leg_l1_joint.dq);
                    atomic_leg_l1.tau.store(leg_l1_joint.tau);
                    break;
                case 2:
                    atomic_leg_l2.q.store(leg_l2_joint.q);
                    atomic_leg_l2.dq.store(leg_l2_joint.dq);
                    atomic_leg_l2.tau.store(leg_l2_joint.tau);
                    break;
                case 3:
                    atomic_leg_l3.q.store(leg_l3_joint.q);
                    atomic_leg_l3.dq.store(leg_l3_joint.dq);
                    atomic_leg_l3.tau.store(leg_l3_joint.tau);
                    break;
                case 4:
                    atomic_leg_l4.q.store(leg_l4_joint.q);
                    atomic_leg_l4.dq.store(leg_l4_joint.dq);
                    atomic_leg_l4.tau.store(leg_l4_joint.tau);
                    break;
                case 5:
                    atomic_leg_l5.q.store(leg_l5_joint.q);
                    atomic_leg_l5.dq.store(leg_l5_joint.dq);
                    atomic_leg_l5.tau.store(leg_l5_joint.tau);
                    break;
            }
        }
        
        // 控制脖子关节 (ID 6)
        cmd.id = 6;
        cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
        cmd.kp = 0.0;
        cmd.kd = 0.0;
        cmd.q = 0.0;
        cmd.dq = 0.0 * queryGearRatio(MotorType::GO_M8010_6);
        cmd.tau = 0.0;
        
        // 发送命令并获取脖子数据（阻塞）
        serialLeft.sendRecv(&cmd, &neck_n1_joint);
        
        // 立即更新脖子原子变量（无锁）
        atomic_neck_n1.q.store(neck_n1_joint.q);
        atomic_neck_n1.dq.store(neck_n1_joint.dq);
        atomic_neck_n1.tau.store(neck_n1_joint.tau);
        
        // 计算频率（每100次循环更新一次）
        loop_count++;
        if (loop_count >= 100) {
            auto current_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_freq_update).count();
            left_motor_frequency = 1000000.0f * loop_count / duration; // 转换为Hz
            loop_count = 0;
            last_freq_update = current_time;
        }
    }
}

// 右腿电机控制线程函数 - 最高频率运行，无锁更新
void rightMotorControlThread() {
    SerialPort serialRight("/dev/ttyright");
    MotorCmd cmd;
    cmd.motorType = MotorType::GO_M8010_6;
    
    // 本地电机数据变量
    MotorData leg_r1_joint, leg_r2_joint, leg_r3_joint, leg_r4_joint, leg_r5_joint;
    MotorData* leg_r_joints[5] = {&leg_r1_joint, &leg_r2_joint, &leg_r3_joint, &leg_r4_joint, &leg_r5_joint};
    
    // 初始化电机类型
    for(int i = 0; i < 5; i++) {
        leg_r_joints[i]->motorType = MotorType::GO_M8010_6;
    }
    
    // 频率测量变量
    auto last_freq_update = std::chrono::steady_clock::now();
    int loop_count = 0;
    
    while (right_motor_thread_running) {
        // 控制右腿5个关节 (ID 1-5) - 阻塞获取数据后立即更新原子变量
        for(int i = 1; i < 6; i++) {
            cmd.id = i;
            cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
            cmd.kp = 0.0;
            cmd.kd = 0.0;
            cmd.q = 0.0;
            cmd.dq = 0.0 * queryGearRatio(MotorType::GO_M8010_6);
            cmd.tau = 0.0;
            
            // 发送命令并获取数据（阻塞）
            serialRight.sendRecv(&cmd, leg_r_joints[i-1]);
            
            // 立即更新原子变量（无锁）
            switch(i) {
                case 1: 
                    atomic_leg_r1.q.store(leg_r1_joint.q);
                    atomic_leg_r1.dq.store(leg_r1_joint.dq);
                    atomic_leg_r1.tau.store(leg_r1_joint.tau);
                    break;
                case 2:
                    atomic_leg_r2.q.store(leg_r2_joint.q);
                    atomic_leg_r2.dq.store(leg_r2_joint.dq);
                    atomic_leg_r2.tau.store(leg_r2_joint.tau);
                    break;
                case 3:
                    atomic_leg_r3.q.store(leg_r3_joint.q);
                    atomic_leg_r3.dq.store(leg_r3_joint.dq);
                    atomic_leg_r3.tau.store(leg_r3_joint.tau);
                    break;
                case 4:
                    atomic_leg_r4.q.store(leg_r4_joint.q);
                    atomic_leg_r4.dq.store(leg_r4_joint.dq);
                    atomic_leg_r4.tau.store(leg_r4_joint.tau);
                    break;
                case 5:
                    atomic_leg_r5.q.store(leg_r5_joint.q);
                    atomic_leg_r5.dq.store(leg_r5_joint.dq);
                    atomic_leg_r5.tau.store(leg_r5_joint.tau);
                    break;
            }
        }
        
        // 计算频率（每100次循环更新一次）
        loop_count++;
        if (loop_count >= 100) {
            auto current_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_freq_update).count();
            right_motor_frequency = 1000000.0f * loop_count / duration; // 转换为Hz
            loop_count = 0;
            last_freq_update = current_time;
        }
    }
}


int main() {
  // 启动所有线程
  std::thread battery_thread(batteryCommThread);
  std::thread log_thread(logDisplayThread);
  std::thread left_motor_thread(leftMotorControlThread);
  std::thread right_motor_thread(rightMotorControlThread);

  // 主线程等待用户中断
  while(true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
  }

  // 程序结束时停止所有线程
  battery_thread_running = false;
  log_thread_running = false;
  left_motor_thread_running = false;
  right_motor_thread_running = false;
  
  if (battery_thread.joinable()) {
    battery_thread.join();
  }
  if (log_thread.joinable()) {
    log_thread.join();
  }
  if (left_motor_thread.joinable()) {
    left_motor_thread.join();
  }
  if (right_motor_thread.joinable()) {
    right_motor_thread.join();
  }

  return 0;
}