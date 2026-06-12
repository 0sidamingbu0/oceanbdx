/*
 * OceanBDX - ONNX policy runner (IsaacLab rsl_rl 导出的 policy.onnx)
 *
 * 观测向量 (与 IsaacLab velocity 任务对齐, 无线速度版本):
 *   [ ang_vel(3) * scale, projected_gravity(3), commands(3) * scale,
 *     (dof_pos - default_dof_pos) * scale, dof_vel * scale, last_actions ]
 * 输出: actions, 目标关节角 = default_dof_pos + action_scale * actions
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OCEANBDX_POLICY_HPP
#define OCEANBDX_POLICY_HPP

#include "oceanbdx/config.hpp"
#include "oceanbdx/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace Ort { struct Env; struct Session; struct MemoryInfo; }

namespace oceanbdx
{

class PolicyRunner
{
public:
    explicit PolicyRunner(const Config &cfg);
    ~PolicyRunner();

    bool Load();          // 加载 ONNX 模型
    void Reset();         // 清零 last_actions

    // 计算一步策略, 返回目标关节角 (URDF坐标, rad)
    // q/dq: 当前关节状态; quat: (w,x,y,z); gyro: rad/s; cmd: (vx,vy,wz)
    std::vector<double> Step(const std::vector<double> &q,
                             const std::vector<double> &dq,
                             const std::array<double, 4> &quat,
                             const std::array<double, 3> &gyro,
                             const std::array<double, 3> &cmd);

    const std::vector<float> &LastActions() const { return last_actions_; }

    // 工具: 四元数(w,x,y,z)旋转重力向量到机体系
    static std::array<double, 3> ProjectedGravity(const std::array<double, 4> &quat);

private:
    Config cfg_;
    std::vector<float> last_actions_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace oceanbdx

#endif // OCEANBDX_POLICY_HPP
