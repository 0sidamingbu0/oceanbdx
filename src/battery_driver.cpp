/*
 * OceanBDX - 电池/BMS 串口驱动实现 (A5 协议)
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/battery_driver.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>

namespace oceanbdx
{

namespace
{
speed_t BaudConst(int baud)
{
    switch (baud)
    {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    default: return B9600;
    }
}

// 查询帧
const uint8_t kQueryPacket[] = {0xA5, 0x40, 0x90, 0x08, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x7D};
// 应答帧头
const uint8_t kReplyHeader[] = {0xA5, 0x01, 0x90, 0x08};
constexpr int kReplyMinLen = 12;
} // namespace

BatteryDriver::BatteryDriver(const std::string &port, int baud) : port_(port), baud_(baud) {}

BatteryDriver::~BatteryDriver() { Stop(); }

bool BatteryDriver::Start()
{
    fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0)
    {
        std::cerr << "[BatteryDriver] open " << port_ << " failed" << std::endl;
        return false;
    }

    struct termios opt;
    tcgetattr(fd_, &opt);
    cfsetispeed(&opt, BaudConst(baud_));
    cfsetospeed(&opt, BaudConst(baud_));
    opt.c_cflag = (opt.c_cflag & ~CSIZE) | CS8;
    opt.c_cflag |= CLOCAL | CREAD;
    opt.c_cflag &= ~(PARENB | PARODD | CSTOPB);
    opt.c_lflag = 0;
    opt.c_oflag = 0;
    opt.c_iflag = 0;
    opt.c_cc[VTIME] = 1; // 100ms 读超时
    opt.c_cc[VMIN] = 0;
    tcflush(fd_, TCIOFLUSH);
    tcsetattr(fd_, TCSANOW, &opt);

    buf_[0] = BatteryState{};
    buf_[1] = BatteryState{};
    front_.store(0);
    running_ = true;
    thread_ = std::thread(&BatteryDriver::CommLoop, this);
    return true;
}

void BatteryDriver::Stop()
{
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0)
    {
        close(fd_);
        fd_ = -1;
    }
}

BatteryState BatteryDriver::GetState() const
{
    BatteryState s = buf_[front_.load(std::memory_order_acquire)];
    s.valid = valid_.load();
    return s;
}

void BatteryDriver::CommLoop()
{
    uint8_t recv[64];
    int update_count = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (running_)
    {
        // 1. 发送查询帧
        if (write(fd_, kQueryPacket, sizeof(kQueryPacket)) != static_cast<ssize_t>(sizeof(kQueryPacket)))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // 2. 等待并读取应答
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ssize_t n = read(fd_, recv, sizeof(recv));

        // 3. 在缓冲中查找应答帧头并解析
        for (ssize_t i = 0; n >= kReplyMinLen && i <= n - kReplyMinLen; ++i)
        {
            if (memcmp(recv + i, kReplyHeader, sizeof(kReplyHeader)) != 0) continue;

            uint16_t cumulative_raw = (recv[i + 4] << 8) | recv[i + 5];
            uint16_t gather_raw = (recv[i + 6] << 8) | recv[i + 7];
            uint16_t current_raw = (recv[i + 8] << 8) | recv[i + 9];
            uint16_t soc_raw = (recv[i + 10] << 8) | recv[i + 11];

            BatteryState st;
            st.cumulative_voltage = cumulative_raw * 0.1;
            st.gather_voltage = gather_raw * 0.1;
            st.current = (static_cast<int>(current_raw) - 30000) * 0.1;
            st.soc = soc_raw * 0.1;
            st.valid = true;

            int back = 1 - front_.load(std::memory_order_relaxed);
            buf_[back] = st;
            front_.store(back, std::memory_order_release);
            valid_.store(true);

            if (++update_count >= 20)
            {
                auto t1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                if (us > 0) update_hz_.store(1e6 * update_count / us);
                update_count = 0;
                t0 = t1;
            }
            break;
        }

        // 4. 20Hz 查询节奏 (已睡10ms, 再补40ms)
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
}

} // namespace oceanbdx
