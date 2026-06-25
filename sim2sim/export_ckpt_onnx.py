#!/usr/bin/env python3
"""把指定 rsl-rl checkpoint (.pt) 导出为 policy.onnx, 与 IsaacLab
export_policy_as_onnx 等价: 归一化 (x-mean)/(std+eps) 并入 MLP 之前,
输入名 'obs' 输出名 'actions', 单 batch。

用法:
  python.sh export_ckpt_onnx.py --ckpt <model_xxxx.pt> --out policy/policy.onnx
"""
import argparse
import os

import torch
from torch import nn


class ActorWithNorm(nn.Module):
    def __init__(self, mean, std, eps, layers):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp((x - self.mean) / (self.std + self.eps))


def build(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["actor_state_dict"]
    mean = a["obs_normalizer._mean"].reshape(1, -1).float()
    std = a["obs_normalizer._std"].reshape(1, -1).float()
    idx = sorted(int(k.split(".")[1]) for k in a if k.startswith("mlp.") and k.endswith(".weight"))
    layers = []
    for n, i in enumerate(idx):
        W = a[f"mlp.{i}.weight"].float()
        b = a[f"mlp.{i}.bias"].float()
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W)
            lin.bias.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    model = ActorWithNorm(mean, std, 1e-2, layers).eval()
    obs_dim = mean.shape[1]
    return model, obs_dim, int(ck.get("iter", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, obs_dim, it = build(args.ckpt)
    dummy = torch.zeros(1, obs_dim)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes=None, opset_version=17,
    )
    # sanity
    import onnxruntime as ort
    import numpy as np
    s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    y = s.run(None, {"obs": np.zeros((1, obs_dim), np.float32)})[0][0]
    print(f"[export] iter={it} -> {args.out}  obs_dim={obs_dim}  zeros->absmax={np.max(np.abs(y)):.3f}")


if __name__ == "__main__":
    main()
