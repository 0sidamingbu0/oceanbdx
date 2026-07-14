/*
 * OceanBDX - USB 手柄驱动实现 (Linux joystick API)
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/gamepad_driver.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/joystick.h>
#include <unistd.h>

namespace oceanbdx
{

namespace
{
constexpr double kAxisMax = 32767.0; // joystick 轴量程 (-32767..32767)
} // namespace

GamepadDriver::GamepadDriver(const std::string &device) : device_(device) {}

GamepadDriver::~GamepadDriver() { Stop(); }

bool GamepadDriver::Start()
{
    fd_ = open(device_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ < 0)
    {
        std::cerr << "[GamepadDriver] open " << device_ << " failed: " << strerror(errno)
                  << " (插好USB手柄, 确认设备为 /dev/input/jsX)" << std::endl;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        buf_[0] = GamepadState{};
        buf_[1] = GamepadState{};
        front_.store(0);
        connected_.store(true);
    }
    running_ = true;
    thread_ = std::thread(&GamepadDriver::ReadLoop, this);
    return true;
}

void GamepadDriver::Stop()
{
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0)
    {
        close(fd_);
        fd_ = -1;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        connected_.store(false);
    }
}

GamepadState GamepadDriver::GetState() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!connected_.load())
    {
        return GamepadState{};
    }
    GamepadState s = buf_[front_.load(std::memory_order_acquire)];
    s.connected = true;
    return s;
}

void GamepadDriver::ReadLoop()
{
    // 本地累积状态, 每收到事件更新后发布到后缓冲
    GamepadState cur;
    js_event ev;

    while (running_)
    {
        ssize_t n = read(fd_, &ev, sizeof(ev));
        if (n == sizeof(ev))
        {
            // JS_EVENT_INIT 位表示上电初始同步事件, 同样需要写入初始值
            switch (ev.type & ~JS_EVENT_INIT)
            {
            case JS_EVENT_AXIS:
                if (ev.number < cur.axes.size())
                    cur.axes[ev.number] = ev.value / kAxisMax;
                break;
            case JS_EVENT_BUTTON:
                if (ev.number < cur.buttons.size())
                    cur.buttons[ev.number] = ev.value ? 1 : 0;
                break;
            default:
                break;
            }

            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                int back = 1 - front_.load(std::memory_order_relaxed);
                buf_[back] = cur;
                front_.store(back, std::memory_order_release);
            }
        }
        else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            // 暂无事件, 短暂休眠避免空转
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        else if (n < 0)
        {
            // 设备被拔出等错误
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                connected_.store(false);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

} // namespace oceanbdx
