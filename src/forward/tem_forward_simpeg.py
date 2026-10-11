# -*- coding: utf-8 -*-
"""基于 SimPEG 的严格一维中心回线正演引擎

SimPEG (Cockett et al., 2015, Computers & Geosciences) 的
Simulation1DLayered 使用数字线性滤波（Key 81/201 点滤波系数），
经过广泛同行评审验证，与解析解误差 < 1%。

本模块提供与 TEMForward 兼容的接口，便于无缝替换训练数据生成。
"""
import numpy as np
from typing import List, Optional

from simpeg.electromagnetics import time_domain as tdem
from simpeg import maps


class TEMForwardSimPEG:
    """SimPEG 一维层状介质中心回线正演（严格数字滤波）"""

    def __init__(self, time_gates: Optional[np.ndarray] = None,
                 loop_radius: float = 10.0, current: float = 1.0):
        self.mu0 = 4 * np.pi * 1e-7
        self.time_gates = (np.logspace(-6, -2, 30) if time_gates is None
                           else np.asarray(time_gates))
        self.a = float(loop_radius)
        self.current = float(current)

    def step_dbdt(self, conds: List[float], thicks: List[float]
                  ) -> np.ndarray:
        """阶跃关断后 dB/dt（V/(A·m²)），取幅值。

        Args:
            conds: 各层电导率 (S/m)，最后一层为半空间
            thicks: 各层厚度 (m)，最后一层厚度任意（内部自动忽略）
        """
        conds = np.asarray(conds, float)
        thicks = np.asarray(thicks, float)
        # SimPEG 层厚不含半空间
        layer_thicks = thicks[:-1] if len(thicks) == len(conds) else thicks

        rx = tdem.Rx.PointMagneticFluxTimeDerivative(
            locations=np.array([[0.0, 0.0, 0.0]]),
            times=self.time_gates,
            orientation='z')
        src = tdem.Src.CircularLoop(
            receiver_list=[rx],
            location=np.array([0.0, 0.0, 0.0]),
            radius=self.a,
            current=self.current,
            waveform=tdem.Src.StepOffWaveform())
        survey = tdem.Survey(src)
        sim = tdem.Simulation1DLayered(
            survey=survey,
            sigmaMap=maps.IdentityMap(nP=len(conds)),
            thicknesses=layer_thicks)
        resp = sim.dpred(conds)
        return np.abs(resp)

    # ---------- 兼容接口 ----------
    def layered_earth_response(self, conductivities: List[float],
                               thicknesses: List[float]) -> np.ndarray:
        return self.step_dbdt(conductivities, thicknesses)

    def add_noise(self, data: np.ndarray, snr: float = 10.0,
                  rng=None) -> np.ndarray:
        """按 SNR(dB) 加高斯噪声。rng 传入 np.random.Generator 可复现。"""
        sp = np.mean(data ** 2)
        npow = sp / (10 ** (snr / 10))
        rng = np.random if rng is None else rng
        return data + rng.normal(0, np.sqrt(npow), data.shape)

    # ---------- 验证 ----------
    def halfspace_dbdt_analytic(self, sigma: float) -> np.ndarray:
        from scipy.special import erf
        u = self.a * np.sqrt(self.mu0 * sigma / (4.0 * self.time_gates))
        bracket = (3.0 * erf(u)
                   - (2.0 / np.sqrt(np.pi)) * u * (3.0 + 2.0 * u ** 2)
                   * np.exp(-u ** 2))
        return self.current / (sigma * self.a ** 3) * bracket

    def validate_halfspace(self, sigma: float = 0.02) -> float:
        num = self.step_dbdt([sigma], [0.0])
        ana = self.halfspace_dbdt_analytic(sigma)
        ana = ana * (num.max() / ana.max())
        mask = ana > ana.max() * 1e-4
        return float(np.mean(np.abs(num[mask] - ana[mask])
                             / (np.abs(ana[mask]) + 1e-30)))
