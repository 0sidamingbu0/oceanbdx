/*
 * OceanBDX - 控制状态机实现
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/fsm.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace oceanbdx
{

const char *FsmStateName(FsmState s)
{
    switch (s)
    {
    case FsmState::PASSIVE: return "PASSIVE";
    case FsmState::BOOT_CHECK: return "BOOT_CHECK";
    case FsmState::SIT_HOLD: return "SIT_HOLD";
    case FsmState::STAND_UP: return "STAND_UP";
    case FsmState::RL_BALANCE: return "RL_BALANCE";
    case FsmState::RL_WALK: return "RL_WALK";
    case FsmState::DAMPING: return "DAMPING";
    }
    return "?";
}

Fsm::Fsm(const Config &cfg, const JointCalibration &calib, PolicyRunner *policy)
    : cfg_(cfg), calib_(calib), policy_(policy)
{
    rl_target_ = cfg_.stand_pose;
}

void Fsm::Transit(FsmState next)
{
    std::cout << "[FSM] " << FsmStateName(state_) << " -> " << FsmStateName(next) << std::endl;
    state_ = next;
    state_time_ = 0.0;
}

bool Fsm::AttitudeUnsafe(const ImuState &imu) const
{
    // roll/pitch 超过 ~60度 触发保护
    auto g = PolicyRunner::ProjectedGravity(imu.quat);
    return imu.valid && g[2] > -0.5;
}

std::vector<JointCommand> Fsm::Update(const RobotState &s,
                                      const std::array<double, 3> &cmd_vel,
                                      FsmEvent event)
{
    state_time_ += cfg_.control_dt;

    // 全局安全: RL/起立阶段姿态异常 -> 阻尼
    if ((state_ == FsmState::STAND_UP || state_ == FsmState::RL_BALANCE ||
         state_ == FsmState::RL_WALK) && AttitudeUnsafe(s.imu))
    {
        message_ = "attitude protect triggered";
        Transit(FsmState::DAMPING);
    }

    // 事件处理
    switch (event)
    {
    case FsmEvent::DAMP:
        if (state_ != FsmState::PASSIVE) Transit(FsmState::DAMPING);
        break;
    case FsmEvent::PASSIVE_REQ:
        Transit(FsmState::PASSIVE);
        break;
    case FsmEvent::BOOT:
        if (state_ == FsmState::PASSIVE || state_ == FsmState::DAMPING) Transit(FsmState::BOOT_CHECK);
        break;
    case FsmEvent::STAND:
        if (state_ == FsmState::SIT_HOLD)
        {
            stand_start_pose_.resize(cfg_.num_joints);
            for (int i = 0; i < cfg_.num_joints; ++i) stand_start_pose_[i] = s.joints[i].q;
            Transit(FsmState::STAND_UP);
        }
        break;
    case FsmEvent::WALK:
        if (state_ == FsmState::RL_BALANCE) Transit(FsmState::RL_WALK);
        break;
    case FsmEvent::BALANCE:
        if (state_ == FsmState::RL_WALK) Transit(FsmState::RL_BALANCE);
        break;
    default:
        break;
    }

    switch (state_)
    {
    case FsmState::PASSIVE: return DoPassive(s);
    case FsmState::BOOT_CHECK: return DoBootCheck(s);
    case FsmState::SIT_HOLD: return DoSitHold(s);
    case FsmState::STAND_UP: return DoStandUp(s);
    case FsmState::RL_BALANCE: return DoRl(s, {0.0, 0.0, 0.0});
    case FsmState::RL_WALK: return DoRl(s, cmd_vel);
    case FsmState::DAMPING: return DoDamping(s);
    }
    return DoPassive(s);
}

std::vector<JointCommand> Fsm::DoPassive(const RobotState &)
{
    return std::vector<JointCommand>(cfg_.num_joints); // 全零
}

std::vector<JointCommand> Fsm::DoBootCheck(const RobotState &s)
{
    std::vector<double> q(cfg_.num_joints);
    for (int i = 0; i < cfg_.num_joints; ++i) q[i] = s.joints[i].q;

    std::vector<int> bad;
    if (calib_.ValidateBootPose(q, &bad))
    {
        message_ = "boot check OK";
        hold_pose_ = q; // 锁定当前坐姿
        Transit(FsmState::SIT_HOLD);
    }
    else if (state_time_ > 2.0)
    {
        message_ = "boot check FAILED, joints out of sit range:";
        for (int j : bad) message_ += " " + cfg_.joint_names[j];
        std::cerr << "[FSM] " << message_ << std::endl;
        Transit(FsmState::PASSIVE);
    }
    return DoPassive(s);
}

std::vector<JointCommand> Fsm::DoSitHold(const RobotState &)
{
    std::vector<JointCommand> cmds(cfg_.num_joints);
    for (int i = 0; i < cfg_.num_joints; ++i)
    {
        cmds[i].q = hold_pose_[i];
        cmds[i].kp = cfg_.fixed_kp[i];
        cmds[i].kd = cfg_.fixed_kd[i];
    }
    return cmds;
}

std::vector<JointCommand> Fsm::DoStandUp(const RobotState &)
{
    // 余弦插值 sit -> stand
    double r = std::min(state_time_ / cfg_.stand_duration, 1.0);
    double a = 0.5 * (1.0 - std::cos(M_PI * r));

    std::vector<JointCommand> cmds(cfg_.num_joints);
    for (int i = 0; i < cfg_.num_joints; ++i)
    {
        cmds[i].q = stand_start_pose_[i] * (1.0 - a) + cfg_.stand_pose[i] * a;
        cmds[i].kp = cfg_.fixed_kp[i];
        cmds[i].kd = cfg_.fixed_kd[i];
    }

    if (r >= 1.0)
    {
        if (policy_)
        {
            policy_->Reset();
            rl_target_ = cfg_.stand_pose;
            rl_tick_ = 0;
            Transit(FsmState::RL_BALANCE);
        }
        // 无策略时停留在站立姿态保持 (调试模式)
    }
    return cmds;
}

std::vector<JointCommand> Fsm::DoRl(const RobotState &s, const std::array<double, 3> &cmd)
{
    if (policy_ && (rl_tick_++ % cfg_.decimation == 0))
    {
        std::vector<double> q(cfg_.num_joints), dq(cfg_.num_joints);
        for (int i = 0; i < cfg_.num_joints; ++i)
        {
            q[i] = s.joints[i].q;
            dq[i] = s.joints[i].dq;
        }
        rl_target_ = policy_->Step(q, dq, s.imu.quat, s.imu.gyro, cmd);
    }

    std::vector<JointCommand> cmds(cfg_.num_joints);
    for (int i = 0; i < cfg_.num_joints; ++i)
    {
        // 软限位
        double qt = std::min(std::max(rl_target_[i], cfg_.joint_lower[i]), cfg_.joint_upper[i]);
        cmds[i].q = qt;
        cmds[i].kp = cfg_.rl_kp[i];
        cmds[i].kd = cfg_.rl_kd[i];
    }
    return cmds;
}

std::vector<JointCommand> Fsm::DoDamping(const RobotState &)
{
    std::vector<JointCommand> cmds(cfg_.num_joints);
    for (int i = 0; i < cfg_.num_joints; ++i) cmds[i].kd = cfg_.damping_kd;
    return cmds;
}

} // namespace oceanbdx
