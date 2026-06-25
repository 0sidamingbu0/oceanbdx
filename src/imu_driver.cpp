/*
 * OceanBDX - YIS320 IMU driver 实现
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/imu_driver.hpp"

#include <cmath>
#include <cstring>
#include <chrono>
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
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 921600: return B921600;
    default: return B460800;
    }
}
} // namespace

ImuDriver::ImuDriver(const std::string &port, int baud) : port_(port), baud_(baud) {}

ImuDriver::~ImuDriver() { Stop(); }

bool ImuDriver::Start()
{
    fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0)
    {
        std::cerr << "[ImuDriver] open " << port_ << " failed" << std::endl;
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
    tcflush(fd_, TCIFLUSH);
    tcsetattr(fd_, TCSANOW, &opt);

    recv_idx_ = 0;
    memset(&info_, 0, sizeof(info_));
    running_ = true;
    thread_ = std::thread(&ImuDriver::ReadLoop, this);
    return true;
}

void ImuDriver::Stop()
{
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0)
    {
        close(fd_);
        fd_ = -1;
    }
}

ImuState ImuDriver::GetState() const
{
    ImuState s;
    const Sample &b = buf_[front_.load(std::memory_order_acquire)];
    for (int i = 0; i < 4; ++i) s.quat[i] = b.quat[i];
    for (int i = 0; i < 3; ++i)
    {
        s.gyro[i] = b.gyro[i];
        s.accel[i] = b.accel[i];
    }
    s.valid = valid_.load();
    return s;
}

void ImuDriver::ReadLoop()
{
    char buf[512];
    int update_count = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (running_)
    {
        int n = read(fd_, buf, sizeof(buf));
        if (n > 0)
        {
            if (recv_idx_ + n > static_cast<int>(sizeof(recv_buf_)))
                recv_idx_ = 0; // 溢出保护
            memcpy(recv_buf_ + recv_idx_, buf, n);
            recv_idx_ += n;
        }

        int pos = 0, cnt = recv_idx_;
        if (cnt < YIS_OUTPUT_MIN_BYTES) continue;

        while (cnt > 0)
        {
            int ret = analysis_data(recv_buf_ + pos, cnt, &info_);
            if (ret == analysis_done)
            {
                pos++;
                cnt--;
            }
            else if (ret == data_len_err)
            {
                break;
            }
            else if (ret == crc_err || ret == analysis_ok)
            {
                output_data_header_t *hdr = (output_data_header_t *)(recv_buf_ + pos);
                int len = hdr->len + YIS_OUTPUT_MIN_BYTES;
                cnt -= len;
                pos += len;

                if (ret == analysis_ok)
                {
                    int back = 1 - front_.load(std::memory_order_relaxed);
                    Sample &b = buf_[back];
                    b.quat[0] = info_.attitude.quaternion_data0; // w
                    b.quat[1] = info_.attitude.quaternion_data1; // x
                    b.quat[2] = info_.attitude.quaternion_data2; // y
                    b.quat[3] = info_.attitude.quaternion_data3; // z
                    b.gyro[0] = info_.angle_rate.x * M_PI / 180.0;
                    b.gyro[1] = info_.angle_rate.y * M_PI / 180.0;
                    b.gyro[2] = info_.angle_rate.z * M_PI / 180.0;
                    b.accel[0] = info_.accel.x;
                    b.accel[1] = info_.accel.y;
                    b.accel[2] = info_.accel.z;
                    front_.store(back, std::memory_order_release);
                    sequence_.fetch_add(1, std::memory_order_release);
                    valid_.store(true);

                    if (++update_count >= 100)
                    {
                        auto t1 = std::chrono::steady_clock::now();
                        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                        update_hz_.store(1e6 * update_count / us);
                        update_count = 0;
                        t0 = t1;
                    }
                }
            }
        }

        memmove(recv_buf_, recv_buf_ + pos, cnt);
        recv_idx_ = cnt;
    }
}

} // namespace oceanbdx
