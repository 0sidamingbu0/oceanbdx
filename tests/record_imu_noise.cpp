/*
 * IMU static noise recorder.
 *
 * Usage:
 *   ./record_imu_noise [config.yaml|/dev/ttyimu] [duration_s] [output.csv] [baud]
 *
 * Records de-duplicated IMU samples while the sensor is static and writes:
 *   t_sec, seq, hz, quat, gyro(rad/s), accel(m/s^2), projected_gravity
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "oceanbdx/config.hpp"
#include "oceanbdx/imu_driver.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct RunningStats
{
    uint64_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();

    void Add(double x)
    {
        ++n;
        const double delta = x - mean;
        mean += delta / static_cast<double>(n);
        const double delta2 = x - mean;
        m2 += delta * delta2;
        min = std::min(min, x);
        max = std::max(max, x);
    }

    double Variance() const { return n > 1 ? m2 / static_cast<double>(n - 1) : 0.0; }
    double Stddev() const { return std::sqrt(Variance()); }
};

std::array<double, 3> ProjectedGravity(const std::array<double, 4> &q)
{
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    return {
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    };
}

bool LooksLikeDevicePath(const std::string &s)
{
    return s.rfind("/dev/", 0) == 0;
}

void PrintStats(const char *name, const std::array<RunningStats, 3> &stats)
{
    std::cout << name << " mean/std/min/max:" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "  [" << i << "] mean=" << stats[i].mean
                  << " std=" << stats[i].Stddev()
                  << " min=" << stats[i].min
                  << " max=" << stats[i].max << std::endl;
    }
}

} // namespace

int main(int argc, char **argv)
{
    std::string config_or_port = (argc > 1) ? argv[1] : "config/oceanbdx.yaml";
    double duration_s = (argc > 2) ? std::atof(argv[2]) : 30.0;
    std::string output_path = (argc > 3) ? argv[3] : "logs/imu_noise_static.csv";
    int baud_override = (argc > 4) ? std::atoi(argv[4]) : 0;

    std::string port;
    int baud = 460800;
    if (LooksLikeDevicePath(config_or_port))
    {
        port = config_or_port;
        if (baud_override > 0) baud = baud_override;
    }
    else
    {
        oceanbdx::Config cfg = oceanbdx::Config::Load(config_or_port);
        port = cfg.imu_port;
        baud = (baud_override > 0) ? baud_override : cfg.imu_baud;
    }

    if (duration_s <= 0.0)
    {
        std::cerr << "duration_s must be positive" << std::endl;
        return 2;
    }

    std::filesystem::path out(output_path);
    if (out.has_parent_path()) std::filesystem::create_directories(out.parent_path());

    std::ofstream csv(output_path);
    if (!csv)
    {
        std::cerr << "failed to open output csv: " << output_path << std::endl;
        return 2;
    }
    csv << "t_sec,seq,imu_hz,"
           "quat_w,quat_x,quat_y,quat_z,"
           "gyro_x,gyro_y,gyro_z,"
           "accel_x,accel_y,accel_z,"
           "proj_g_x,proj_g_y,proj_g_z\n";
    csv << std::fixed << std::setprecision(9);

    oceanbdx::ImuDriver imu(port, baud);
    if (!imu.Start()) return 1;

    std::cout << "Recording IMU static noise: port=" << port
              << " baud=" << baud
              << " duration_s=" << duration_s
              << " output=" << output_path << std::endl;
    std::cout << "Keep the IMU completely still and level until recording finishes." << std::endl;

    std::array<RunningStats, 3> gyro_stats;
    std::array<RunningStats, 3> accel_stats;
    std::array<RunningStats, 3> proj_g_stats;
    std::array<RunningStats, 4> quat_stats;

    auto t_start = std::chrono::steady_clock::now();
    auto last_print = t_start;
    uint64_t last_seq = 0;
    uint64_t samples = 0;

    while (true)
    {
        auto now = std::chrono::steady_clock::now();
        const double t_sec = std::chrono::duration<double>(now - t_start).count();
        if (t_sec >= duration_s) break;

        const uint64_t seq = imu.Sequence();
        if (seq != last_seq && imu.IsValid())
        {
            last_seq = seq;
            auto s = imu.GetState();
            auto g = ProjectedGravity(s.quat);

            csv << t_sec << "," << seq << "," << imu.UpdateHz();
            for (double v : s.quat) csv << "," << v;
            for (double v : s.gyro) csv << "," << v;
            for (double v : s.accel) csv << "," << v;
            for (double v : g) csv << "," << v;
            csv << "\n";

            for (int i = 0; i < 4; ++i) quat_stats[i].Add(s.quat[i]);
            for (int i = 0; i < 3; ++i)
            {
                gyro_stats[i].Add(s.gyro[i]);
                accel_stats[i].Add(s.accel[i]);
                proj_g_stats[i].Add(g[i]);
            }
            ++samples;
        }

        if (std::chrono::duration<double>(now - last_print).count() >= 1.0)
        {
            last_print = now;
            std::cout << "  t=" << std::fixed << std::setprecision(1) << t_sec
                      << "s samples=" << samples
                      << " imu_hz=" << std::setprecision(1) << imu.UpdateHz()
                      << (imu.IsValid() ? " valid" : " waiting") << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }

    imu.Stop();
    csv.close();

    std::cout << "\nDone. samples=" << samples << " csv=" << output_path << std::endl;
    if (samples < 10)
    {
        std::cerr << "Too few samples; check IMU port, baud, and whether another process owns the serial device." << std::endl;
        return 1;
    }

    std::cout << std::setprecision(10);
    PrintStats("gyro_rad_s", gyro_stats);
    PrintStats("accel_m_s2", accel_stats);
    PrintStats("projected_gravity", proj_g_stats);

    std::cout << "quat mean/std/min/max:" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "  [" << i << "] mean=" << quat_stats[i].mean
                  << " std=" << quat_stats[i].Stddev()
                  << " min=" << quat_stats[i].min
                  << " max=" << quat_stats[i].max << std::endl;
    }

    std::cout << "\nTraining noise seed values:" << std::endl;
    std::cout << "  gyro_std_rad_s: ["
              << gyro_stats[0].Stddev() << ", "
              << gyro_stats[1].Stddev() << ", "
              << gyro_stats[2].Stddev() << "]" << std::endl;
    std::cout << "  projected_gravity_std: ["
              << proj_g_stats[0].Stddev() << ", "
              << proj_g_stats[1].Stddev() << ", "
              << proj_g_stats[2].Stddev() << "]" << std::endl;
    std::cout << "  accel_std_m_s2: ["
              << accel_stats[0].Stddev() << ", "
              << accel_stats[1].Stddev() << ", "
              << accel_stats[2].Stddev() << "]" << std::endl;

    return 0;
}
