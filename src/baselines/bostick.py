# -*- coding: utf-8 -*-
"""Bostick 变换近似反演（经典基线方法）

原理（中心回线晚期近似）：
    探测深度  d(t) = sqrt(rho_a(t) * t / (2*pi*mu0_inv))，mu0_inv = 1e7/(4*pi)
    电阻率    rho_B = rho_a * (1 + s) / (1 - s)，s = d(ln rho_a)/d(ln t)
其中 rho_a 为全时间视电阻率曲线。本实现输入为归一化响应曲线，
先按幅值映射为视电阻率代理曲线，再做 Bostick 变换，输出深度-电阻率剖面。
导数用 Savitzky-Golay 平滑求取以压制噪声放大。

参考：Bostick (1977)；刘国兴等《电法勘探原理与方法》。
"""
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple


class BostickInverter:
    def __init__(self, time_gates: np.ndarray = None):
        self.mu0 = 4 * np.pi * 1e-7
        self.time_gates = (np.logspace(-6, -2, 30) if time_gates is None
                           else np.asarray(time_gates))

    def _to_apparent_resistivity(self, response: np.ndarray,
                                 rho_ref: float = 100.0) -> np.ndarray:
        """将归一化响应幅值映射为视电阻率代理曲线（保持单调映射与量纲）"""
        r = np.abs(np.asarray(response, float))
        r = np.maximum(r, r.max() * 1e-6 + 1e-30)
        # 晚期响应大→浅层高阻；响应小→导电层。取反比并锚定到背景电阻率
        rho_a = rho_ref * (r / np.median(r)) ** -1.0
        return np.maximum(rho_a, 1e-2)

    def invert(self, response: np.ndarray, rho_ref: float = 100.0
               ) -> Tuple[np.ndarray, np.ndarray]:
        """单点反演。

        Args:
            response: (n_time,) 归一化响应
            rho_ref: 背景电阻率锚定值 (Ω·m)

        Returns:
            (depths, rho): 深度 (m) 与 Bostick 电阻率 (Ω·m)，按深度升序
        """
        t = self.time_gates
        rho_a = self._to_apparent_resistivity(response, rho_ref)

        # 对数域斜率 s = d(ln rho_a)/d(ln t)，SG 平滑抑制导数噪声
        lt, lr = np.log(t), np.log(rho_a)
        win = min(7, len(t) // 2 * 2 - 1)
        if win >= 5:
            lr_s = savgol_filter(lr, win, 2, deriv=1, delta=lt[1] - lt[0])
        else:
            lr_s = np.gradient(lr, lt)
        s = np.clip(lr_s, -0.95, 0.95)  # 限制斜率避免 (1+s)/(1-s) 发散

        depths = np.sqrt(rho_a * t / (2 * np.pi * self.mu0))
        rho_b = rho_a * (1 + s) / (1 - s)

        order = np.argsort(depths)
        return depths[order], rho_b[order]

    def invert_profile_grid(self, response: np.ndarray,
                            depth_grid: np.ndarray,
                            rho_ref: float = 100.0) -> np.ndarray:
        """反演并插值到固定深度网格，便于与其他方法逐点对比"""
        d, r = self.invert(response, rho_ref)
        return np.interp(depth_grid, d, r, left=r[0], right=r[-1])
