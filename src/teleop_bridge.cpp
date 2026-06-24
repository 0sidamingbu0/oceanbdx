/*
 * OceanBDX - 真机遥操作桥接进程 (sim2sim 联调用)
 *
 * 复用部署同款 LegDriver (宇树 GO-M8010-6, 每腿一路 USB-485, 独立线程跑满
 * 总线原生频率 ~116Hz, 无 Python GIL 干扰), 通过本地 UDP 与 MuJoCo (Python)
 * 交换关节命令/状态:
 *   - Python -> 本进程: 目标关节角(URDF) + kp + kd + enable
 *   - 本进程 -> Python: 关节读数 q/dq/tau(URDF) + 在线掩码
 *
 * 坐标换算 (URDF<->电机输出轴) 与 src/main.cpp 完全一致 (JointCalibration)。
 * LegDriver 内部再做减速比换算到转子侧。
 *
 * 安全: UDP 接收超时 (Python 退出/卡死) 自动给双腿上阻尼, 防止保持旧的高 kp 设定值。
 *
 * 用法:
 *   ./oceanbdx_teleop [config/oceanbdx.yaml] [udp_port=9090]
 * ★ 与 oceanbdx_run 互斥 (争抢同一串口), 联调时只能运行其一。
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/calibration.hpp"
#include "oceanbdx/config.hpp"
#include "oceanbdx/leg_driver.hpp"
#include "oceanbdx/types.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

using namespace oceanbdx;

namespace
{
constexpr uint32_t kCmdMagic = 0x4F424358;   // 'OBCX' (Python -> C++)
constexpr uint32_t kStateMagic = 0x4F425358; // 'OBSX' (C++ -> Python)

std::atomic<bool> g_running{true};
void OnSignal(int) { g_running = false; }
} // namespace

int main(int argc, char **argv)
{
    const std::string config_path = (argc > 1) ? argv[1] : "config/oceanbdx.yaml";
    const int port = (argc > 2) ? std::atoi(argv[2]) : 9090;

    Config cfg = Config::Load(config_path);
    JointCalibration calib(cfg);
    const int nj = cfg.num_joints;

    signal(SIGINT, OnSignal);
    signal(SIGTERM, OnSignal);

    // ---- 驱动 (与 main.cpp 同款) ----
    LegDriver left("left", cfg.left_leg, cfg.gear_ratio);
    LegDriver right("right", cfg.right_leg, cfg.gear_ratio);
    if (!left.Start() || !right.Start())
    {
        std::cerr << "[teleop] leg driver start failed (serial busy? stop oceanbdx_run first)"
                  << std::endl;
        return -1;
    }

    // 关节下标 -> (腿, 腿内下标)
    struct Map { LegDriver *leg; int idx; };
    std::vector<Map> jmap(nj, {nullptr, -1});
    auto map_joint = [&](const char *leg_name, LegDriver *leg, int joint_index, int leg_index) {
        if (joint_index < 0 || joint_index >= nj)
        {
            std::cerr << "[teleop] invalid " << leg_name << " joint index " << joint_index << std::endl;
            return false;
        }
        if (leg_index < 0 || leg_index >= leg->NumMotors())
        {
            std::cerr << "[teleop] invalid " << leg_name << " motor slot " << leg_index << std::endl;
            return false;
        }
        if (jmap[joint_index].leg != nullptr)
        {
            std::cerr << "[teleop] duplicate joint mapping for index " << joint_index << std::endl;
            return false;
        }
        jmap[joint_index] = {leg, leg_index};
        return true;
    };
    bool mapping_ok = true;
    for (size_t k = 0; k < cfg.left_leg.joint_indices.size(); ++k)
        mapping_ok &= map_joint("left", &left, cfg.left_leg.joint_indices[k], static_cast<int>(k));
    for (size_t k = 0; k < cfg.right_leg.joint_indices.size(); ++k)
        mapping_ok &= map_joint("right", &right, cfg.right_leg.joint_indices[k], static_cast<int>(k));
    for (int i = 0; i < nj; ++i)
    {
        if (jmap[i].leg == nullptr || jmap[i].idx < 0)
        {
            std::cerr << "[teleop] invalid joint mapping for index " << i << " ("
                      << (i < static_cast<int>(cfg.joint_names.size()) ? cfg.joint_names[i] : "?")
                      << ")" << std::endl;
            mapping_ok = false;
        }
    }
    if (!mapping_ok)
    {
        left.SetDamping(cfg.damping_kd);
        right.SetDamping(cfg.damping_kd);
        left.Stop();
        right.Stop();
        return -1;
    }

    // ---- UDP socket (本机回环) ----
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        std::cerr << "[teleop] socket() failed" << std::endl;
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        std::cerr << "[teleop] bind(:" << port << ") failed (port in use?)" << std::endl;
        close(sock);
        return -1;
    }
    // 接收超时 100ms: 超时即认为 Python 断开, 给电机上阻尼
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::cout << "[teleop] ready on udp 127.0.0.1:" << port
              << " (nj=" << nj << ", gear=" << cfg.gear_ratio << ")" << std::endl;

    std::vector<uint8_t> rbuf(4096), sbuf(4096);
    std::vector<double> q(nj), kp(nj), kd(nj);
    bool peer_known = false;
    sockaddr_in peer{};
    socklen_t peer_len = sizeof(peer);
    int print_count = 0;

    while (g_running)
    {
        ssize_t n = recvfrom(sock, rbuf.data(), rbuf.size(), 0,
                             reinterpret_cast<sockaddr *>(&peer), &peer_len);

        bool valid = false;
        uint32_t enable = 0;
        if (n >= static_cast<ssize_t>(12))
        {
            uint32_t magic = 0, pkt_n = 0;
            std::memcpy(&magic, rbuf.data() + 0, 4);
            std::memcpy(&enable, rbuf.data() + 4, 4);
            std::memcpy(&pkt_n, rbuf.data() + 8, 4);
            const size_t need = 12 + static_cast<size_t>(pkt_n) * 3 * sizeof(double);
            if (magic == kCmdMagic && pkt_n == static_cast<uint32_t>(nj) &&
                n >= static_cast<ssize_t>(need))
            {
                size_t off = 12;
                std::memcpy(q.data(), rbuf.data() + off, nj * sizeof(double));
                off += nj * sizeof(double);
                std::memcpy(kp.data(), rbuf.data() + off, nj * sizeof(double));
                off += nj * sizeof(double);
                std::memcpy(kd.data(), rbuf.data() + off, nj * sizeof(double));
                valid = true;
                peer_known = true;
            }
        }

        if (valid)
        {
            // 写命令 (URDF -> 电机输出轴). enable=0 时 kp=kd=0 (可手动搬动)
            for (int i = 0; i < nj; ++i)
            {
                JointCommand mc;
                if (enable)
                {
                    mc.q = calib.UrdfToMotor(i, q[i]);
                    mc.kp = kp[i];
                    mc.kd = kd[i];
                }
                else
                {
                    mc.kp = 0.0;
                    mc.kd = 0.0;
                }
                mc.dq = 0.0;
                mc.tau = 0.0;
                jmap[i].leg->SetCommand(jmap[i].idx, mc);
            }
        }
        else if (n < 0)
        {
            // 接收超时: Python 断开 -> 阻尼保护
            left.SetDamping(cfg.damping_kd);
            right.SetDamping(cfg.damping_kd);
        }

        // 回发状态 (电机输出轴 -> URDF)
        if (peer_known)
        {
            uint32_t online = 0;
            std::memcpy(sbuf.data() + 0, &kStateMagic, 4);
            uint32_t nj_u = static_cast<uint32_t>(nj);
            std::memcpy(sbuf.data() + 4, &nj_u, 4);
            const size_t q_off = 12;
            const size_t dq_off = q_off + nj * sizeof(double);
            const size_t tau_off = dq_off + nj * sizeof(double);
            for (int i = 0; i < nj; ++i)
            {
                JointState ms = jmap[i].leg->GetState(jmap[i].idx);
                double sq = calib.MotorToUrdf(i, ms.q);
                double sdq = calib.MotorToUrdfVel(i, ms.dq);
                double stau = calib.MotorToUrdfTau(i, ms.tau);
                std::memcpy(sbuf.data() + q_off + i * sizeof(double), &sq, sizeof(double));
                std::memcpy(sbuf.data() + dq_off + i * sizeof(double), &sdq, sizeof(double));
                std::memcpy(sbuf.data() + tau_off + i * sizeof(double), &stau, sizeof(double));
                if (jmap[i].leg->IsOnline(jmap[i].idx))
                    online |= (1u << i);
            }
            std::memcpy(sbuf.data() + 8, &online, 4);
            const size_t slen = 12 + static_cast<size_t>(nj) * 3 * sizeof(double);
            sendto(sock, sbuf.data(), slen, 0,
                   reinterpret_cast<sockaddr *>(&peer), peer_len);
        }

        if (++print_count >= 200)
        {
            print_count = 0;
            std::cout << "[teleop] left_hz=" << static_cast<int>(left.LoopHz())
                      << " right_hz=" << static_cast<int>(right.LoopHz())
                      << " enable=" << enable << std::endl;
        }
    }

    std::cout << "[teleop] shutting down, damping..." << std::endl;
    left.SetDamping(cfg.damping_kd);
    right.SetDamping(cfg.damping_kd);
    usleep(200000);
    left.Stop();
    right.Stop();
    close(sock);
    return 0;
}
