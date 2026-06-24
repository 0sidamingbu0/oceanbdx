/*
 * OceanBDX - 控制状态机
 *
 * PASSIVE ──(L1/'0')──▶ SIT_ALIGN(脚本回蹲姿) ──校验通过──▶ SIT_HOLD ──(start/'1')──▶
 * STAND_UP(脚本起立) ──完成──▶ RL_BALANCE(站立策略) ──(A/'2')──▶ RL_WALK(定速行走)
 *
 * 任何状态按 select/'9' 或姿态保护触发 ──▶ DAMPING (阻尼软停)
 * RL_WALK 速度指令归零自动回 RL_BALANCE 行为 (同一policy时仅指令不同)。
 *
 * 设计说明 (启动→站立 用脚本而不是RL):
 *   - 起立轨迹是确定性大范围姿态变化, 用固定时长的余弦插值脚本最安全可控;
 *   - RL策略只在站立附近的状态分布内训练 (自平衡/行走), 起立完成后再切入RL;
 *   - 若以后想做RL起立, 只需新增一个状态复用同一接口。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_FSM_HPP
#define OCEANBDX_FSM_HPP

#include "oceanbdx/calibration.hpp"
#include "oceanbdx/config.hpp"
#include "oceanbdx/policy.hpp"
#include "oceanbdx/types.hpp"

#include <functional>
#include <string>
#include <vector>

namespace oceanbdx
{

enum class FsmState
{
    PASSIVE,     // 上电默认: 全部电机零增益
    BOOT_CHECK,  // 坐姿校验电机绝对位置 (多圈歧义检查)
    SIT_ALIGN,   // 缓慢移动到坐姿/蹲姿, 然后做坐姿校验
    SIT_HOLD,    // 坐姿位置保持
    STAND_UP,    // 脚本插值: 坐姿 -> 站立姿态
    RL_BALANCE,  // RL自平衡站立 (cmd = 0)
    RL_WALK,     // RL定速行走 (cmd = 手柄/固定速度)
    DAMPING,     // 阻尼保护 (kd-only 软停)
};

const char *FsmStateName(FsmState s);

// 事件由键盘/手柄映射
enum class FsmEvent
{
    NONE,
    BOOT,        // 请求缓慢回坐姿并校验
    STAND,       // 请求起立
    WALK,        // 进入行走
    BALANCE,     // 回到静态平衡
    DAMP,        // 软急停
    PASSIVE_REQ, // 回到完全失能
};

class Fsm
{
public:
    Fsm(const Config &cfg, const JointCalibration &calib, PolicyRunner *policy);

    // 每个控制周期调用一次。
    // state: 当前机器人状态 (URDF坐标); cmd_vel: 速度指令;
    // 返回每个关节的命令 (URDF坐标)。
    std::vector<JointCommand> Update(const RobotState &state,
                                     const std::array<double, 3> &cmd_vel,
                                     FsmEvent event);

    FsmState State() const { return state_; }
    const std::string &Message() const { return message_; }
    const std::vector<double> &RlTarget() const { return rl_target_; }

private:
    void Transit(FsmState next);
    bool AttitudeUnsafe(const ImuState &imu) const;

    std::vector<JointCommand> DoPassive(const RobotState &s);
    std::vector<JointCommand> DoBootCheck(const RobotState &s);
    std::vector<JointCommand> DoSitAlign(const RobotState &s);
    std::vector<JointCommand> DoSitHold(const RobotState &s);
    std::vector<JointCommand> DoStandUp(const RobotState &s);
    std::vector<JointCommand> DoRl(const RobotState &s, const std::array<double, 3> &cmd);
    std::vector<JointCommand> DoDamping(const RobotState &s);

    Config cfg_;
    JointCalibration calib_;
    PolicyRunner *policy_;

    FsmState state_ = FsmState::PASSIVE;
    std::string message_;
    double state_time_ = 0.0;                 // 当前状态持续时间 (s)
    int rl_tick_ = 0;                         // decimation 计数
    std::vector<double> sit_align_start_pose_; // 回蹲姿插值起点
    std::vector<double> stand_start_pose_;    // 起立插值起点
    std::vector<double> hold_pose_;           // SIT_HOLD 锁定姿态
    std::vector<double> rl_target_;           // 最近一次策略输出
};

} // namespace oceanbdx

#endif // OCEANBDX_FSM_HPP
