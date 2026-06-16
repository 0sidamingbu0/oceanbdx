#!/usr/bin/env python3
"""
OceanBDX 真机电机桥接 (sim2sim 联调用) —— UDP 客户端

真正驱动电机的是 C++ 进程 oceanbdx_teleop (复用部署同款 LegDriver,
跑满总线原生频率 ~116Hz, 无 Python GIL 干扰)。本模块只是它的瘦客户端:
  - 后台线程按 poll_hz 把当前目标(URDF角+kp+kd+enable)用 UDP 发给 C++ 桥接
  - 收回真机读数 q/dq/tau(URDF) + 在线掩码, 供 MuJoCo 镜像显示

对外 API 与原来一致 (start/stop/set_enabled/set_target/get_state/online_mask),
因此 mujoco_sim.py 无需改动。

★ 安全:
  - C++ 桥接与 oceanbdx_run 互斥 (争抢串口); 联调前先停掉真机主控。
  - 若本客户端停止发包, C++ 端 100ms 超时自动给电机上阻尼。

SPDX-License-Identifier: Apache-2.0
"""
import os
import socket
import struct
import subprocess
import threading
import time

import numpy as np

CMD_MAGIC = 0x4F424358    # 'OBCX'  Python -> C++
STATE_MAGIC = 0x4F425358  # 'OBSX'  C++ -> Python

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealBridge:
    """真机腿部电机桥接 (UDP 客户端). 线程安全, 对外全部 URDF 坐标。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = int(cfg.get("num_joints", 10))

        real = cfg.get("real", {})
        self.port = int(real.get("udp_port", 9090))
        self.host = real.get("udp_host", "127.0.0.1")
        self.poll_hz = float(real.get("poll_hz", 200))
        self.auto_launch = bool(real.get("auto_launch", True))
        self.bridge_exec = real.get(
            "bridge_exec", os.path.join(ROOT, "build/oceanbdx_teleop"))
        self.config_path = real.get(
            "config_path", os.path.join(ROOT, "config/oceanbdx.yaml"))

        self._lock = threading.Lock()
        self._tgt_q = np.zeros(self.n)
        self._tgt_kp = np.zeros(self.n)
        self._tgt_kd = np.zeros(self.n)
        self._enabled = False

        self._st_q = np.zeros(self.n)
        self._st_dq = np.zeros(self.n)
        self._st_tau = np.zeros(self.n)
        self._online = np.zeros(self.n, dtype=bool)

        self._sock = None
        self._proc = None
        self._thread = None
        self._running = False

    # ---------- 生命周期 ----------
    def start(self):
        # 1) 可选: 自动拉起 C++ 桥接进程
        if self.auto_launch:
            if not os.path.exists(self.bridge_exec):
                print(f"[real] 找不到 C++ 桥接可执行文件: {self.bridge_exec}\n"
                      f"       请先编译: cd build && cmake .. && make oceanbdx_teleop")
                return False
            try:
                self._proc = subprocess.Popen(
                    [self.bridge_exec, self.config_path, str(self.port)])
                time.sleep(0.5)  # 等待串口打开 / bind
            except Exception as e:
                print(f"[real] 启动 C++ 桥接失败: {e}")
                return False
            if self._proc.poll() is not None:
                print("[real] C++ 桥接进程已退出 (串口被占用? 先停掉 oceanbdx_run)")
                return False
        else:
            print(f"[real] 假定 oceanbdx_teleop 已在 {self.host}:{self.port} 运行")

        # 2) UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.2)

        # 3) 握手: 发一帧 disabled 命令, 等首个状态回包确认链路
        if not self._handshake():
            print("[real] 未收到 C++ 桥接的状态回包, 握手失败")
            self.stop()
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[real] UDP 桥接已连接 {self.host}:{self.port} (nj={self.n})")
        return True

    def _handshake(self):
        for _ in range(20):  # 最多等 ~2s
            self._send_cmd(np.zeros(self.n), np.zeros(self.n),
                           np.zeros(self.n), False)
            try:
                self._recv_state()
                return True
            except (socket.timeout, OSError):
                time.sleep(0.1)
        return False

    def stop(self):
        self.set_enabled(False)
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        # 多发几帧 disabled, 确保电机回到自由/阻尼
        if self._sock:
            for _ in range(3):
                try:
                    self._send_cmd(np.zeros(self.n), np.zeros(self.n),
                                   np.zeros(self.n), False)
                except OSError:
                    break
                time.sleep(0.02)
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._sock:
            self._sock.close()
            self._sock = None
        print("[real] 桥接已停止")

    # ---------- 命令 / 状态 ----------
    def set_enabled(self, on):
        with self._lock:
            self._enabled = bool(on)

    def set_target(self, q_urdf, kp, kd):
        q = np.asarray(q_urdf, dtype=float)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        with self._lock:
            self._tgt_q[:len(q)] = q
            self._tgt_kp[:len(kp)] = kp
            self._tgt_kd[:len(kd)] = kd

    def get_state(self):
        with self._lock:
            return self._st_q.copy(), self._st_dq.copy(), self._st_tau.copy()

    def online_mask(self):
        with self._lock:
            return self._online.copy()

    # ---------- UDP 编解码 ----------
    def _send_cmd(self, q, kp, kd, enable):
        hdr = struct.pack("<III", CMD_MAGIC, 1 if enable else 0, self.n)
        body = (q.astype("<f8").tobytes() + kp.astype("<f8").tobytes()
                + kd.astype("<f8").tobytes())
        self._sock.sendto(hdr + body, (self.host, self.port))

    def _recv_state(self):
        data, _ = self._sock.recvfrom(4096)
        if len(data) < 12:
            raise OSError("short packet")
        magic, n, online = struct.unpack("<III", data[:12])
        if magic != STATE_MAGIC or n != self.n:
            raise OSError("bad state packet")
        off = 12
        cnt = self.n
        q = np.frombuffer(data, "<f8", cnt, off)
        dq = np.frombuffer(data, "<f8", cnt, off + cnt * 8)
        tau = np.frombuffer(data, "<f8", cnt, off + 2 * cnt * 8)
        with self._lock:
            self._st_q[:] = q
            self._st_dq[:] = dq
            self._st_tau[:] = tau
            for i in range(cnt):
                self._online[i] = bool(online & (1 << i))

    # ---------- 后台循环 (心跳: 持续发命令防止 C++ 端超时阻尼) ----------
    def _loop(self):
        dt = 1.0 / self.poll_hz
        while self._running:
            t0 = time.time()
            with self._lock:
                q = self._tgt_q.copy()
                kp = self._tgt_kp.copy()
                kd = self._tgt_kd.copy()
                en = self._enabled
            try:
                self._send_cmd(q, kp, kd, en)
                self._recv_state()
            except (socket.timeout, OSError):
                pass
            left = dt - (time.time() - t0)
            if left > 0:
                time.sleep(left)
