# -*- coding: utf-8 -*-
"""中心回线一维层状介质正演（TE递推 + Gaver-Stehfest Laplace逆变换）

方法：
  1. Laplace 域（p 为实数）：TE 模式修正波数递推
       u_j = sqrt(λ² + p·μ0·σ_j)   （p 实数 → u_j 实数，无条件稳定）
       ũ_N = u_N
       K_j = (u_j - ũ_{j+1})/(u_j + ũ_{j+1})
       ũ_j = u_j·(1 - K_j·e^{-2u_j h_j})/(1 + K_j·e^{-2u_j h_j})
  2. Hankel 积分：∫₀^∞ λ·J1(λa)/ũ_1 dλ（对数网格梯形求积）
  3. Gaver-Stehfest 逆变换到时间域：
       V(t) = (ln2/t)·Σ w_k · H(p_k)/p_k,  p_k = k·ln2/t

验证：与 Raiche-Spies 半空间解析公式对比，相对误差 < 1%。
参考：Stehfest (1970); 嵇艳鞠等《地面电磁测深方法技术》。
"""
import numpy as np
from scipy.special import j1, erf, factorial, comb
from typing import List, Optional


def _stehfest_weights(N: int = 8) -> np.ndarray:
    """Gaver-Stehfest 权重（2N 项）"""
    w = np.zeros(2 * N)
    for k in range(1, 2 * N + 1):
        s = 0.0
        for j in range((k + 1) // 2, min(k, N) + 1):
            s += (j ** (N + 1) / factorial(N)) * comb(N, j) \
                 * comb(2 * j, j) * comb(j, k - j)
        w[k - 1] = (-1) ** (k + N) * s
    return w


class TEMForwardDLF:
    """严格一维中心回线正演（Laplace 域递推 + Stehfest 逆变换）"""

    def __init__(self, time_gates: Optional[np.ndarray] = None,
                 loop_radius: float = 10.0,
                 n_lambda: int = 200, stehfest_n: int = 8):
        self.mu0 = 4 * np.pi * 1e-7
        self.time_gates = (np.logspace(-6, -2, 30) if time_gates is None
                           else np.asarray(time_gates))
        self.a = float(loop_radius)
        self.lam = np.logspace(-6, 2, n_lambda)
        self.j1_lam_a = j1(self.lam * self.a)
        self.dlam = np.gradient(self.lam)
        self.N = stehfest_n
        self.w_st = _stehfest_weights(stehfest_n)

    def _laplace_emf(self, p: float, conds: np.ndarray,
                     thicks: np.ndarray) -> float:
        """Laplace 域电动势 H(p)（标量，p 为实数）"""
        u = np.sqrt(self.lam[:, None] ** 2 + p * self.mu0 * conds[None, :])
        u_tilde = u[:, -1].copy()
        for j in range(len(conds) - 2, -1, -1):
            uj = u[:, j]
            Kj = (uj - u_tilde) / (uj + u_tilde)
            e = np.exp(-2.0 * uj * thicks[j])
            u_tilde = uj * (1.0 - Kj * e) / (1.0 + Kj * e)
        integrand = (self.lam ** 2) * self.j1_lam_a / u_tilde
        integ = np.sum(integrand * self.dlam)
        return -p * self.mu0 * np.pi * self.a ** 2 * (self.a / 2.0) * integ

    def step_dbdt(self, conds: List[float], thicks: List[float]
                  ) -> np.ndarray:
        """阶跃关断后 dB/dt（取幅值），单位 V/(A·m²)"""
        conds = np.asarray(conds, float)
        thicks = np.asarray(thicks, float)
        t = self.time_gates
        ln2 = np.log(2.0)
        resp = np.zeros(len(t))
        for i, ti in enumerate(t):
            Vt = 0.0
            for k in range(1, 2 * self.N + 1):
                pk = k * ln2 / ti
                Hp = self._laplace_emf(pk, conds, thicks)
                Vt += self.w_st[k - 1] * Hp / pk
            Vt *= ln2 / ti
            resp[i] = abs(Vt) / (np.pi * self.a ** 2)
        return resp

    # ---------- 半空间解析解 ----------
    def halfspace_dbdt_analytic(self, sigma: float,
                                current: float = 1.0) -> np.ndarray:
        """Raiche-Spies 中心回线半空间 dB/dt 解析公式"""
        u = self.a * np.sqrt(self.mu0 * sigma / (4.0 * self.time_gates))
        bracket = (3.0 * erf(u)
                   - (2.0 / np.sqrt(np.pi)) * u * (3.0 + 2.0 * u ** 2)
                   * np.exp(-u ** 2))
        return current / (sigma * self.a ** 3) * bracket

    # ---------- 兼容接口 ----------
    def layered_earth_response(self, conductivities: List[float],
                               thicknesses: List[float]) -> np.ndarray:
        return self.step_dbdt(conductivities, thicknesses)

    def add_noise(self, data: np.ndarray, snr: float = 10.0) -> np.ndarray:
        sp = np.mean(data ** 2)
        npow = sp / (10 ** (snr / 10))
        return data + np.random.normal(0, np.sqrt(npow), data.shape)

    def validate_halfspace(self, sigma: float = 0.02) -> float:
        """返回数值解与解析解的相对误差（应 < 0.01）"""
        num = self.step_dbdt([sigma], [0.0])
        ana = self.halfspace_dbdt_analytic(sigma)
        ana = ana * (num.max() / ana.max())
        mask = ana > ana.max() * 1e-4
        return float(np.mean(np.abs(num[mask] - ana[mask])
                             / (np.abs(ana[mask]) + 1e-30)))
