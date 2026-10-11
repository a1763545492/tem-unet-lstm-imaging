# -*- coding: utf-8 -*-
"""纯 PyTorch 可微的严格一维中心回线 TEM 正演（与 SimPEG 物理一致）

用于物理约束损失（PhysicsInformedLoss）：网络预测的电导率剖面经本
可微正演重算 dB/dt 响应，梯度可穿过正演算子回传至网络。

实现路线（与 simpeg 0.25 Simulation1DLayered 完全同源）：
  1) 与模型无关的系数（Hankel 滤波 λ 点、C1 几何系数、频率道、
     余弦频时投影矩阵 A）由 SimPEG 预计算一次，注册为 torch buffer；
  2) 与模型有关的部分——TE 模式反射系数 rTE 的 tanh 递推
     （geoana rTE_forward 公式，Ward & Hohmann, 1988）——用纯
     torch 复数运算重写，对 σ 严格可微；
  3) Hankel DLF（key_101_2009，J1 核）+ 余弦变换（key_81_2009）
     均为矩阵乘法，天然可微。

磁导率统一取 μ0（非磁性地层），导纳中的 1/(iωμ0) 公因子可约去，
rTE 直接用 u = sqrt(λ² + iωμ0σ) 递推，减少复数除法。

典型规模：n_lambda=101, n_freq=128, n_time=30；
u 的中间张量 (B, n_layer, n_freq, n_lambda) 较大，
提供 freq_chunk 参数按频率分块以降低显存/内存峰值。
"""
import numpy as np
import torch
import torch.nn as nn

MU0 = 4.0 * np.pi * 1e-7


def _build_simpeg_coefficients(time_gates: np.ndarray,
                               thicknesses: np.ndarray,
                               loop_radius: float,
                               current: float):
    """构建 SimPEG 1D 仿真并提取与模型无关的预计算系数。

    Returns:
        freqs: (n_freq,) Hz
        lambs: (n_lambda,) 空间波数 1/m
        c1:    (n_lambda,) 中心回线 Hankel 几何系数（C0=0）
        j1:    (n_lambda,) J1 数字滤波权重
        a_time:(n_time, n_freq) 余弦频时变换+样条插值矩阵（已含 μ0）
    """
    from simpeg.electromagnetics import time_domain as tdem
    from simpeg import maps

    time_gates = np.asarray(time_gates, float)
    n_layers = len(thicknesses) + 1
    rx = tdem.Rx.PointMagneticFluxTimeDerivative(
        locations=np.array([[0.0, 0.0, 0.0]]),
        times=time_gates, orientation='z')
    src = tdem.Src.CircularLoop(
        receiver_list=[rx], location=np.array([0.0, 0.0, 0.0]),
        radius=float(loop_radius), current=float(current),
        waveform=tdem.Src.StepOffWaveform())
    survey = tdem.Survey([src])
    sim = tdem.Simulation1DLayered(
        survey=survey,
        sigmaMap=maps.IdentityMap(nP=n_layers),
        thicknesses=np.asarray(thicknesses, float))
    sim._compute_coefficients()
    # 中心水平回线、中心点 z 分量接收：C0s 恒为 0
    assert np.allclose(sim._C0s, 0.0), '中心回线配置下 C0 应为 0'
    return (sim._frequencies.astype(np.float64),                 # Hz
            sim._unique_lambs.astype(np.float64),
            sim._C1s[0].astype(np.float64),
            sim._fhtfilt.j1.astype(np.float64),
            sim._As[0].astype(np.float64))


class TorchDLFTEMForward(nn.Module):
    """与 SimPEG 一致的可微 1D 层状中心回线 dB/dt 正演。

    Args:
        time_gates: 时间道 (s)
        thicknesses: 各层厚度 (m)，长度 n_layer-1，末层为半空间
        loop_radius/current: 回线半径 (m) 与发射电流 (A)
        freq_chunk: 频率分块大小，降低中间张量内存峰值（None=不分块）
    """

    def __init__(self, time_gates: np.ndarray,
                 thicknesses: np.ndarray,
                 loop_radius: float = 10.0,
                 current: float = 1.0,
                 freq_chunk: int = 64):
        super().__init__()
        freqs, lambs, c1, j1, a_time = _build_simpeg_coefficients(
            time_gates, thicknesses, loop_radius, current)
        self.n_layer = len(thicknesses) + 1
        self.freq_chunk = freq_chunk
        self.register_buffer('freqs', torch.tensor(freqs, dtype=torch.float64))
        self.register_buffer('lambs', torch.tensor(lambs, dtype=torch.float64))
        self.register_buffer('c1', torch.tensor(c1, dtype=torch.float64))
        self.register_buffer('j1', torch.tensor(j1, dtype=torch.float64))
        self.register_buffer('a_time',
                             torch.tensor(a_time, dtype=torch.float64))
        self.register_buffer('thicks',
                             torch.tensor(np.asarray(thicknesses, float),
                                          dtype=torch.float64))

    def _rte_hankel(self, sigma: torch.Tensor, f_idx: slice) -> torch.Tensor:
        """计算指定频率段的频域 Hankel 变换结果。

        Args:
            sigma: (B, n_layer) 实数电导率 S/m
            f_idx: 频率段切片
        Returns:
            v_real: (B, n_freq_chunk) 频域响应实部
        """
        freqs = self.freqs[f_idx]                       # (F,)
        omega = 2.0 * np.pi * freqs                    # (F,)
        lambs = self.lambs                             # (K,)
        thicks = self.thicks                           # (n_layer-1,)

        # k² = -iωμ0σ ；u = sqrt(λ² - k²) = sqrt(λ² + iωμ0σ)
        # 形状：(B, n_layer, F, K)
        wms = torch.complex(torch.zeros_like(omega),
                            omega * MU0)                   # (F,) complex
        kterm = wms[None, None, :, None] * sigma[:, :, None, None]
        l2 = lambs[None, None, None, :] ** 2
        u = torch.sqrt(l2 + kterm)                     # 实部为正，无跨支切问题

        # μ 统一为 μ0 时 TE 导纳 Y=u/(iωμ0)，公因子在 rTE 中约去，
        # 直接以 u 作为导纳递推（geoana tanh 形式）。
        tanh = torch.tanh(u[:, :-1] * thicks[None, :, None, None])
        zh = u[:, -1]                                  # (B, F, K) 半空间
        for k in range(self.n_layer - 2, -1, -1):
            uk = u[:, k]
            tk = tanh[:, k]
            zh = uk * (zh + uk * tk) / (uk + zh * tk)
        rte = (lambs - zh) / (lambs + zh)             # (B, F, K)

        # Hankel DLF：v(ω) = Σ_k C1(λ_k) rTE(λ_k,ω) · w^J1_k
        v = torch.sum(rte * self.c1[None, None, :]
                      * self.j1[None, None, :], dim=-1)  # (B, F)
        return v.real

    def forward(self, sigma_profile: torch.Tensor) -> torch.Tensor:
        """sigma_profile: (B, n_layer) 电导率 (S/m) → |dB/dt| (B, T)

        内部以 float64 复数递推保证数值稳定（物理项只占总损失一小部分，
        样本数通常子采样，计算量可控）。
        """
        orig_dtype = sigma_profile.dtype
        sigma = sigma_profile.double()
        n_freq = self.freqs.shape[0]
        chunk = self.freq_chunk or n_freq
        v_chunks = []
        for s in range(0, n_freq, chunk):
            v_chunks.append(self._rte_hankel(sigma, slice(s, s + chunk)))
        v_real = torch.cat(v_chunks, dim=1)            # (B, n_freq)
        # 频时余弦变换 + 时间道插值；A 已含 μ0（dB/dt 接收点）
        dbdt = v_real @ self.a_time.t()                # (B, n_time)
        return torch.abs(dbdt).to(orig_dtype)
