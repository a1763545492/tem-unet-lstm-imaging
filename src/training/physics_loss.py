# -*- coding: utf-8 -*-
"""创新点A：物理约束混合损失（Physics-Informed Loss）

核心思想：除标签监督损失外，将网络预测的电导率剖面经**可微正演算子**
重算响应，与实测（输入）响应比较，形成正演一致性约束：
    L = λ1·L_data + λ2·L_physics + λ3·L_smooth
- L_data   : MSE(预测模型, 标签模型)
- L_physics: MSE(log10(F(预测电导率)), log10(输入响应))，F 为严格 1D TEM
             正演（Hankel DLF + 余弦变换），由外部注入 forward_op
             （src/forward/tem_forward_torch.py 的 TorchDLFTEMForward，
             与 SimPEG 数字滤波正演数值一致，误差 <1e-7）
- L_smooth : 电导率剖面一阶差分平滑正则（稳定解、抑制跳变噪声）

可微性：网络输出（log10σ）经 10**x 保正还原，梯度可穿透 DLF 正演算子
回传至网络 —— 实现真正的物理引导训练。

数值保护：物理项在对数幅值空间度量（响应跨数量级，线性 MSE 不稳定）；
严格 DLF 反向传播开销大，每 batch 仅子采样 physics_batch 个样本参与。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """物理约束混合损失。

    forward(pred, target, input_data, input_stats)
    - pred:        (B, 2, L) 网络输出（channel0=电导率剖面）
    - target:      (B, 2, L) 标签
    - input_data:  (B, 1, T) 网络输入（归一化响应）
    - input_stats: (mu, sd) 各 (B,)，用于把输入反归一化回物理幅值
    """

    def __init__(self, forward_op,
                 lambda_data: float = 1.0, lambda_physics: float = 0.3,
                 lambda_smooth: float = 0.05, adaptive: bool = True,
                 physics_batch: int = 8):
        super().__init__()
        self.fwd = forward_op
        self.l1, self.l2, self.l3 = lambda_data, lambda_physics, lambda_smooth
        self.adaptive = adaptive
        # 严格 DLF 正演的复数 rTE 递推反向传播开销大，
        # 每个训练 batch 仅子采样若干样本参与物理项，降低训练耗时
        self.physics_batch = physics_batch

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                input_data: torch.Tensor,
                input_stats: tuple,
                apply_physics: bool = True,
                physics_scale: float = 1.0) -> tuple:
        l_data = F.mse_loss(pred, target)

        # 预测电导率：网络输出为 log10(σ)，指数还原保正
        log_sigma = torch.clamp(pred[:, 0, :], -6.0, 1.0)

        # 输入响应反归一化到 log10(物理幅值)
        mu, sd = input_stats                                     # (B,)

        # 电导率剖面平滑正则（对数空间，全样本，计算廉价始终保留）
        l_smooth = torch.mean((log_sigma[:, 1:] - log_sigma[:, :-1]) ** 2)

        if apply_physics:
            # 物理项子采样（确定性取前 N 个，shuffle 的 DataLoader 已保证多样性）
            nb = self.physics_batch or pred.shape[0]
            nb = min(nb, pred.shape[0])
            sigma_pred = 10.0 ** log_sigma[:nb]              # (nb,L)
            resp_pred = self.fwd(sigma_pred)                 # (nb,T)
            d_obs = (input_data.squeeze(1)[:nb] * sd[:nb].unsqueeze(1)
                     + mu[:nb].unsqueeze(1))
            # 物理损失在对数幅值空间（响应跨数量级，线性 MSE 不稳定）
            l_physics = F.mse_loss(torch.log10(resp_pred + 1e-12), d_obs)
        else:
            # 间隔步跳过昂贵的 DLF 正演反向传播
            l_physics = torch.zeros((), device=pred.device, dtype=pred.dtype)

        # 自适应权重：物理/平滑项保持在数据项的 λ 比例，防止量纲失衡
        if not apply_physics:
            w2 = 0.0
        elif self.adaptive:
            w2 = (self.l2 * physics_scale
                  * l_data.detach() / (l_physics.detach() + 1e-12))
        else:
            w2 = self.l2 * physics_scale
        if self.adaptive:
            w3 = self.l3 * l_data.detach() / (l_smooth.detach() + 1e-12)
        else:
            w3 = self.l3

        total = l_data + w2 * l_physics + w3 * l_smooth
        return total, {'data': float(l_data.detach()),
                       'physics': float(l_physics.detach()),
                       'smooth': float(l_smooth.detach())}
