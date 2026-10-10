# -*- coding: utf-8 -*-
"""OCCAM 风格 1D 光滑约束反演（经典基线方法）

最小构造准则：min ||d - F(m)||² + mu * ||R·m||²
- 模型 m = log10(各层电导率)，固定层厚（对数加密网格）
- 正演 F 复用 TEMForward.layered_earth_response（对数空间残差）
- Jacobian 用有限差分（层数 ~20-30，正演快，可接受）
- 正则化因子 mu 自适应递减，Gauss-Newton 迭代

参考：Constable et al. (1987) Occam's inversion。
"""
import numpy as np
from typing import Optional, Tuple

from ..forward.tem_forward import TEMForward


class Occam1D:
    def __init__(self, time_gates: np.ndarray = None,
                 n_layers: int = 30, max_depth: float = 300.0,
                 mu_init: float = 10.0, mu_decay: float = 1.5,
                 max_iter: int = 15, tol: float = 1e-3):
        self.time_gates = (np.logspace(-6, -2, 30) if time_gates is None
                           else np.asarray(time_gates))
        self.fwd = TEMForward(self.time_gates)
        self.n_layers = n_layers
        # 对数加密层厚（浅薄深厚），最后一层厚度置 0（半空间）
        edges = np.concatenate(([0], np.logspace(np.log10(2.0),
                                np.log10(max_depth), n_layers)))
        self.thicknesses = np.diff(edges)
        self.thicknesses[-1] = 0.0
        self.depth_grid = edges[:-1] + np.maximum(np.diff(edges), 1e-3) / 2
        self.depth_grid[-1] = max_depth  # 半空间节点
        self.mu_init, self.mu_decay = mu_init, mu_decay
        self.max_iter, self.tol = max_iter, tol

    def _forward_log(self, m: np.ndarray) -> np.ndarray:
        """对数空间正演：m = log10(sigma) → log10(|response| + eps)"""
        conds = 10.0 ** m
        resp = self.fwd.layered_earth_response(conds.tolist(),
                                               self.thicknesses.tolist())
        return np.log10(np.abs(resp) + 1e-12)

    def _roughness(self) -> np.ndarray:
        R = np.eye(self.n_layers)
        for i in range(self.n_layers - 1):
            R[i, i], R[i, i + 1] = 1.0, -1.0
        return R

    def invert(self, response: np.ndarray,
               sigma0: float = 0.05,
               verbose: bool = False
               ) -> Tuple[np.ndarray, np.ndarray, list]:
        """单点 1D 反演。

        Args:
            response: (n_time,) 响应曲线
            sigma0: 均匀半空间初始电导率 (S/m)
            verbose: 打印迭代信息

        Returns:
            (depth_grid, conductivity, rms_history)
        """
        d_obs = np.log10(np.abs(response) + 1e-12)
        m = np.full(self.n_layers, np.log10(sigma0))
        R = self._roughness()
        mu, rms_hist, prev_rms = self.mu_init, [], np.inf
        eps_fd = 1e-3  # 有限差分步长（对数空间）

        for it in range(self.max_iter):
            f_m = self._forward_log(m)
            rms = float(np.sqrt(np.mean((d_obs - f_m) ** 2)))
            rms_hist.append(rms)
            if verbose:
                print(f'  iter {it}: RMS={rms:.4f}, mu={mu:.2f}')
            if abs(prev_rms - rms) < self.tol:
                break
            prev_rms = rms

            # 有限差分 Jacobian (n_time × n_layers)
            J = np.zeros((len(d_obs), self.n_layers))
            for j in range(self.n_layers):
                m_p = m.copy()
                m_p[j] += eps_fd
                J[:, j] = (self._forward_log(m_p) - f_m) / eps_fd

            # (J'J + mu R'R) dm = J'(d - F(m))
            A = J.T @ J + mu * (R.T @ R)
            b = J.T @ (d_obs - f_m)
            try:
                dm = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dm = np.linalg.lstsq(A, b, rcond=None)[0]
            m = m + np.clip(dm, -1.0, 1.0)  # 限制步长防跳变
            mu = max(mu / self.mu_decay, 1e-2)

        return self.depth_grid, 10.0 ** m, rms_hist

    def invert_profile_grid(self, response: np.ndarray,
                            depth_grid: np.ndarray,
                            sigma0: float = 0.05) -> np.ndarray:
        """反演并插值到固定深度网格（便于与其他方法对比）"""
        d, c, _ = self.invert(response, sigma0)
        return np.interp(depth_grid, d, c, left=c[0], right=c[-1])
