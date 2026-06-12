/*
 * OceanBDX - 单腿驱动实现 (宇树 GO-M8010-6, 一路USB-485)
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/leg_driver.hpp"

#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

#include <chrono>
#include <iostream>

namespace oceanbdx
{

LegDriver::LegDriver(const std::string &name, const LegPortConfig &cfg, double gear_ratio)
    : name_(name), port_name_(cfg.port),
      motor_ids_(cfg.motor_ids), gear_ratio_(gear_ratio)
{
    for (size_t i = 0; i < motor_ids_.size(); ++i)
    {
        cmds_.emplace_back(new AtomicCmd());
        states_.emplace_back(new AtomicState());
        online_.emplace_back(new std::atomic<bool>(false));
    }
}

LegDriver::~LegDriver() { Stop(); }

bool LegDriver::Start()
{
    try
    {
        serial_.reset(new SerialPort(port_name_));
    }
    catch (const std::exception &e)
    {
        std::cerr << "[LegDriver:" << name_ << "] open " << port_name_
                  << " failed: " << e.what() << std::endl;
        return false;
    }
    running_ = true;
    thread_ = std::thread(&LegDriver::PollLoop, this);
    return true;
}

void LegDriver::Stop()
{
    running_ = false;
    if (thread_.joinable()) thread_.join();
    serial_.reset();
}

void LegDriver::SetCommand(int idx, const JointCommand &cmd)
{
    auto &c = *cmds_[idx];
    c.q.store(cmd.q);
    c.dq.store(cmd.dq);
    c.kp.store(cmd.kp);
    c.kd.store(cmd.kd);
    c.tau.store(cmd.tau);
}

JointState LegDriver::GetState(int idx) const
{
    JointState s;
    const auto &st = *states_[idx];
    s.q = st.q.load();
    s.dq = st.dq.load();
    s.tau = st.tau.load();
    return s;
}

void LegDriver::SetDamping(double kd)
{
    for (size_t i = 0; i < cmds_.size(); ++i)
    {
        JointCommand c;
        c.kd = kd;
        SetCommand(static_cast<int>(i), c);
    }
}

void LegDriver::PollLoop()
{
    MotorCmd cmd;
    MotorData data;
    cmd.motorType = MotorType::GO_M8010_6;
    data.motorType = MotorType::GO_M8010_6;

    const double gr = gear_ratio_;
    int loop_count = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (running_)
    {
        for (size_t i = 0; i < motor_ids_.size(); ++i)
        {
            const auto &c = *cmds_[i];
            cmd.id = motor_ids_[i];
            cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
            // SDK 的 q/dq/kp/kd 是转子侧:
            //   q_rotor = q_out * gr,  dq_rotor = dq_out * gr
            //   tau_out = tau_rotor * gr
            //   kp_rotor = kp_out / gr^2, kd_rotor = kd_out / gr^2
            cmd.q = c.q.load() * gr;
            cmd.dq = c.dq.load() * gr;
            cmd.kp = c.kp.load() / (gr * gr);
            cmd.kd = c.kd.load() / (gr * gr);
            cmd.tau = c.tau.load() / gr;

            if (serial_->sendRecv(&cmd, &data))
            {
                auto &s = *states_[i];
                s.q.store(data.q / gr);
                s.dq.store(data.dq / gr);
                s.tau.store(data.tau * gr);
                online_[i]->store(true);
            }
            else
            {
                online_[i]->store(false);
            }
        }

        if (++loop_count >= 100)
        {
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            loop_hz_.store(1e6 * loop_count / us);
            loop_count = 0;
            t0 = t1;
        }
    }

    // 退出前发一帧零增益命令
    for (size_t i = 0; i < motor_ids_.size(); ++i)
    {
        cmd.id = motor_ids_[i];
        cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
        cmd.q = 0; cmd.dq = 0; cmd.kp = 0; cmd.kd = 0; cmd.tau = 0;
        serial_->sendRecv(&cmd, &data);
    }
}

} // namespace oceanbdx
