# -*- coding: utf-8 -*-
"""创新点A：物理约束混合损失（Physics-Informed Loss）

核心思想：除标签监督损失外，将网络预测的电导率剖面经**可微正演算子**
重算响应，与实测（输入）响应比较，形成正演一致性约束：
    L = λ1·L_data + λ2·L_physics + λ3·L_smooth
- L_data   : MSE(预测模型, 标签模型)
- L_physics: MSE(F(预测电导率), 输入响应)，F 为 TEMForward 的 torch 可微复现
- L_smooth : 电导率剖面一阶差分平滑正则（稳定解、抑制跳变噪声）

可微性：网络输出经 softplus 保证电导率为正；erf/exp/sqrt 均为 torch
原生可微算子，梯度可穿透正演算子回传至网络 —— 实现真正的物理引导训练。

数值保护：sqrt(x+eps)、响应幅值在样本内归一化（与输入预处理一致）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableTEMForward(nn.Module):
    """TEMForward.layered_earth_response 的 PyTorch 可微批量化复现。

    将网络输出的 L 节点电导率剖面视为 L 个固定层厚 dz 的地层，
    各层对响应的贡献为 Δσ_i · layer_resp(σ_i, dz)，Δσ 相对最深节点。
    """

    def __init__(self, time_gates: np.ndarray, layer_thickness: float = 10.0):
        super().__init__()
        self.register_buffer('time_gates',
                             torch.tensor(time_gates, dtype=torch.float32))
        self.mu0 = 4 * np.pi * 1e-7
        self.dz = float(layer_thickness)

    def layer_response(self, sigma: torch.Tensor,
                       thickness: float) -> torch.Tensor:
        """sigma: (B, L) → response: (B, L, T)"""
        t = self.time_gates  # (T,)
        tau = self.mu0 * sigma.unsqueeze(-1) * t / 4.0          # (B,L,T)
        sqrt_tau = torch.sqrt(tau + 1e-30)
        erf_val = torch.special.erf(sqrt_tau)
        resp = (2.0 / np.sqrt(np.pi)) * torch.exp(-tau) / sqrt_tau - erf_val
        if thickness > 0:
            k = torch.sqrt(self.mu0 * sigma.unsqueeze(-1) / (4.0 * t) + 1e-30)
            depth_factor = torch.exp(-2.0 * k * thickness)
            resp = resp * (1.0 - depth_factor)
        return resp

    def forward(self, sigma_profile: torch.Tensor) -> torch.Tensor:
        """sigma_profile: (B, L) 电导率(S/m) → 响应 (B, T)（幅值非负）"""
        sigma_base = sigma_profile[:, -1:]                       # (B,1)
        delta = sigma_profile - sigma_base                       # (B,L)
        resp_layers = self.layer_response(sigma_profile, self.dz)  # (B,L,T)
        return torch.abs(torch.sum(delta.unsqueeze(-1) * resp_layers, dim=1))


class PhysicsInformedLoss(nn.Module):
    """物理约束混合损失。

    forward(pred, target, input_data, input_stats)
    - pred:        (B, 2, L) 网络输出（channel0=电导率剖面）
    - target:      (B, 2, L) 标签
    - input_data:  (B, 1, T) 网络输入（归一化响应）
    - input_stats: (mu, sd) 各 (B,)，用于把输入反归一化回物理幅值
    """

    def __init__(self, forward_op: DifferentiableTEMForward,
                 lambda_data: float = 1.0, lambda_physics: float = 0.3,
                 lambda_smooth: float = 0.05, adaptive: bool = True):
        super().__init__()
        self.fwd = forward_op
        self.l1, self.l2, self.l3 = lambda_data, lambda_physics, lambda_smooth
        self.adaptive = adaptive

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                input_data: torch.Tensor,
                input_stats: tuple) -> tuple:
        l_data = F.mse_loss(pred, target)

        # 预测电导率：网络输出为 log10(σ)，指数还原保正
        log_sigma = torch.clamp(pred[:, 0, :], -6.0, 1.0)
        sigma_pred = 10.0 ** log_sigma                           # (B,L)
        resp_pred = self.fwd(sigma_pred)                         # (B,T)

        # 输入响应反归一化到物理幅值
        mu, sd = input_stats                                     # (B,)
        d_obs = input_data.squeeze(1) * sd.unsqueeze(1) + mu.unsqueeze(1)

        # 物理损失在对数幅值空间计算（响应跨数量级，线性 MSE 不稳定）
        l_physics = F.mse_loss(torch.log10(resp_pred + 1e-12),
                               torch.log10(torch.abs(d_obs) + 1e-12))

        # 电导率剖面平滑正则（对数空间）
        l_smooth = torch.mean((log_sigma[:, 1:] - log_sigma[:, :-1]) ** 2)

        # 自适应权重：物理/平滑项保持在数据项的 λ 比例，防止量纲失衡
        if self.adaptive:
            w2 = self.l2 * l_data.detach() / (l_physics.detach() + 1e-12)
            w3 = self.l3 * l_data.detach() / (l_smooth.detach() + 1e-12)
        else:
            w2, w3 = self.l2, self.l3

        total = l_data + w2 * l_physics + w3 * l_smooth
        return total, {'data': float(l_data.detach()),
                       'physics': float(l_physics.detach()),
                       'smooth': float(l_smooth.detach())}
