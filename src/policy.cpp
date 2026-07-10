/*
 * OceanBDX - ONNX policy runner 实现 (onnxruntime)
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/policy.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace oceanbdx
{

struct PolicyRunner::Impl
{
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "oceanbdx"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::string input_name, output_name;
    int64_t obs_dim = 0;
    double gait_phase = 0.0;
};

PolicyRunner::PolicyRunner(const Config &cfg) : cfg_(cfg), impl_(new Impl())
{
    last_actions_.assign(cfg_.num_joints, 0.0f);
    last_raw_actions_.assign(cfg_.num_joints, 0.0f);
}

PolicyRunner::~PolicyRunner() = default;

bool PolicyRunner::Load()
{
    try
    {
        impl_->opts.SetIntraOpNumThreads(2);
        impl_->session.reset(new Ort::Session(impl_->env, cfg_.policy_path.c_str(), impl_->opts));

        Ort::AllocatorWithDefaultOptions alloc;
        impl_->input_name = impl_->session->GetInputNameAllocated(0, alloc).get();
        impl_->output_name = impl_->session->GetOutputNameAllocated(0, alloc).get();
        auto shape = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        impl_->obs_dim = shape.back();
        const int64_t legacy_obs_dim = 11 + 3 * cfg_.num_joints;
        if (cfg_.num_obs > 0 && impl_->obs_dim != cfg_.num_obs)
        {
            std::cerr << "[Policy] model obs_dim=" << impl_->obs_dim
                      << " != config policy.num_obs=" << cfg_.num_obs << std::endl;
            impl_->session.reset();
            return false;
        }
        if (impl_->obs_dim != legacy_obs_dim)
        {
            std::cerr << "[Policy] C++ runner currently constructs only the legacy "
                      << legacy_obs_dim << "-D observation, but the model requires "
                      << impl_->obs_dim
                      << ". Use the Python sim2sim runner until path-frame, base velocity, "
                         "neck state, and 14-D action support are ported."
                      << std::endl;
            impl_->session.reset();
            return false;
        }
        std::cout << "[Policy] loaded " << cfg_.policy_path
                  << " obs_dim=" << impl_->obs_dim << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Policy] load failed: " << e.what() << std::endl;
        return false;
    }
}

void PolicyRunner::Reset()
{
    std::fill(last_actions_.begin(), last_actions_.end(), 0.0f);
    std::fill(last_raw_actions_.begin(), last_raw_actions_.end(), 0.0f);
    impl_->gait_phase = 0.0;
}

std::array<double, 3> PolicyRunner::ProjectedGravity(const std::array<double, 4> &q)
{
    // projected_gravity = quat_rotate_inverse(q, (0,0,-1)), q = (w,x,y,z)
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    std::array<double, 3> g;
    g[0] = -2.0 * (x * z - w * y);
    g[1] = -2.0 * (y * z + w * x);
    g[2] = -(1.0 - 2.0 * (x * x + y * y));
    return g;
}

std::vector<double> PolicyRunner::Step(const std::vector<double> &q,
                                       const std::vector<double> &dq,
                                       const std::array<double, 4> &quat,
                                       const std::array<double, 3> &gyro,
                                       const std::array<double, 3> &cmd)
{
    const int nj = cfg_.num_joints;
    std::vector<float> obs;
    obs.reserve(11 + 3 * nj);

    for (int i = 0; i < 3; ++i)
        obs.push_back(static_cast<float>(gyro[i] * cfg_.ang_vel_scale));

    auto g = ProjectedGravity(quat);
    for (int i = 0; i < 3; ++i)
        obs.push_back(static_cast<float>(g[i]));

    for (int i = 0; i < 3; ++i)
        obs.push_back(static_cast<float>(cmd[i] * cfg_.commands_scale[i]));

    constexpr double kPi = 3.14159265358979323846;
    obs.push_back(static_cast<float>(std::sin(2.0 * kPi * impl_->gait_phase)));
    obs.push_back(static_cast<float>(std::cos(2.0 * kPi * impl_->gait_phase)));
    const double policy_dt = cfg_.control_dt * std::max(1, cfg_.decimation);
    impl_->gait_phase = std::fmod(
        impl_->gait_phase + policy_dt / std::max(1.0e-6, cfg_.gait_cycle_period),
        1.0
    );

    for (int i = 0; i < nj; ++i)
        obs.push_back(static_cast<float>((q[i] - cfg_.default_dof_pos[i]) * cfg_.dof_pos_scale));

    for (int i = 0; i < nj; ++i)
        obs.push_back(static_cast<float>(dq[i] * cfg_.dof_vel_scale));

    for (int i = 0; i < nj; ++i)
        obs.push_back(last_actions_[i]);

    // clip obs
    const float co = static_cast<float>(cfg_.clip_obs);
    for (auto &v : obs) v = std::min(std::max(v, -co), co);

    if (impl_->obs_dim > 0 && static_cast<int64_t>(obs.size()) != impl_->obs_dim)
    {
        std::cerr << "[Policy] obs size " << obs.size() << " != model obs_dim "
                  << impl_->obs_dim << std::endl;
    }

    std::array<int64_t, 2> in_shape{1, static_cast<int64_t>(obs.size())};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        impl_->mem_info, obs.data(), obs.size(), in_shape.data(), in_shape.size());

    const char *in_names[] = {impl_->input_name.c_str()};
    const char *out_names[] = {impl_->output_name.c_str()};
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);

    const float *act = outputs[0].GetTensorData<float>();
    const float ca = static_cast<float>(cfg_.clip_actions);

    std::vector<double> target(nj);
    for (int i = 0; i < nj; ++i)
    {
        last_raw_actions_[i] = act[i];
        float a = std::min(std::max(act[i], -ca), ca);
        last_actions_[i] = a;
        target[i] = cfg_.default_dof_pos[i] + cfg_.action_scale * a;
    }
    return target;
}

} // namespace oceanbdx
