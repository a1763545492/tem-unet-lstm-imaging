# -*- coding: utf-8 -*-
"""TEM 正演兼容入口（严格 SimPEG 数字滤波引擎）

历史说明：本模块原为启发式 erf 叠加近似正演，与 SimPEG 数据生成协议
存在物理不一致（层状响应近似误差可达 10%-33%）。现已统一：TEMForward
为 TEMForwardSimPEG 的兼容薄封装，全部链路（GUI 演示、自动标注、
数据集生成、物理约束损失）共用同一严格 DLF 正演。

数据生成协议（与 scripts/train_physics.py make_dataset 一致）：
- 30 网格阶梯剖面（dz=10 m，max_depth=300 m），三层随机模型
- 电导率 10^[-2.5, -0.7] S/m，界面深度 30-120 m，噪声 SNR 15-30 dB
- 标签：Y[0]=log10σ 剖面，Y[1]=归一化真实厚度剖面（界面处跳变）
"""
import numpy as np
from typing import List, Optional, Tuple

from .tem_forward_simpeg import TEMForwardSimPEG


class TEMForward(TEMForwardSimPEG):
    """严格 1D TEM 正演（SimPEG DLF），兼容旧 TEMForward API。

    继承 TEMForwardSimPEG 的 step_dbdt / layered_earth_response /
    add_noise / halfspace 解析验证，并补充旧版工具方法
    （多类型噪声、合成数据集生成、数据存取）。
    """

    def __init__(self, time_gates: Optional[np.ndarray] = None,
                 loop_radius: float = 10.0, current: float = 1.0):
        super().__init__(time_gates=time_gates, loop_radius=loop_radius,
                         current=current)

    def add_noise(self, data: np.ndarray, snr: float = 10.0,
                  noise_type: str = 'gaussian', rng=None) -> np.ndarray:
        """按信噪比（dB）加噪，保留旧版多噪声类型选项。

        rng: np.random.Generator 或 None（None 时用全局 np.random，
        保持旧行为；传入种子化 Generator 可获得逐位可复现噪声）。
        """
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        rng = np.random if rng is None else rng

        if noise_type == 'gaussian':
            noise = rng.normal(0, np.sqrt(noise_power), data.shape)
        elif noise_type == 'impulse':
            noise = np.zeros_like(data)
            impulse_idx = rng.choice(len(data), int(len(data) * 0.05))
            noise[impulse_idx] = rng.normal(
                0, 3 * np.sqrt(noise_power), len(impulse_idx))
        elif noise_type == 'periodic':
            t = np.arange(len(data))
            noise = (np.sqrt(noise_power) * np.sin(2 * np.pi * t / 5)
                     * rng.normal(0, 1, len(data)))
        else:
            noise = rng.normal(0, np.sqrt(noise_power), data.shape)

        return data + noise

    def generate_synthetic_data(self, n_samples: int = 1000,
                                max_depth: float = 300.0, nz: int = 30,
                                seed: int = 42,
                                snr_range: Tuple[float, float] = (15.0, 30.0),
                                sigma_range: Tuple[float, float] = (-2.5, -0.7),
                                interface_range: Tuple[float, float] = (30.0, 120.0),
                                return_params: bool = False
                                ) -> Tuple[np.ndarray, ...]:
        """新协议合成数据集：30 网格阶梯剖面（与训练/物理损失自洽）。

        Returns:
            data:   (n_samples, T) 含噪 dB/dt 响应（物理幅值，未归一化）
            labels: (n_samples, 2, nz) [log10σ 剖面; 厚度/max_depth]
            params: 仅 return_params=True 时附带，逐样本层参数字典
                    （conductivities/thicknesses/snr/interface_depths）
        """
        rng = np.random.default_rng(seed)
        dz = max_depth / nz
        grid_thicks = np.r_[np.full(nz - 1, dz), 0.0].tolist()
        data = np.zeros((n_samples, self.time_gates.size), dtype=np.float64)
        labels = np.zeros((n_samples, 2, nz), dtype=np.float64)
        params = []
        d = np.arange(nz) * dz
        with np.errstate(all='ignore'):
            for i in range(n_samples):
                conds = 10 ** rng.uniform(*sigma_range, 3)
                ths = [rng.uniform(*interface_range),
                       rng.uniform(*interface_range), 0.0]
                b = np.cumsum(ths)[:2]
                cond_prof = np.where(d < b[0], conds[0],
                                     np.where(d < b[1], conds[1], conds[2]))
                thick_prof = np.where(d < b[0], ths[0],
                                      np.where(d < b[1], ths[1], 0.0))
                resp = self.layered_earth_response(cond_prof.tolist(),
                                                   grid_thicks)
                snr = float(rng.uniform(*snr_range))
                resp = self.add_noise(resp, snr=snr, rng=rng)
                data[i] = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
                labels[i, 0] = np.log10(cond_prof)
                labels[i, 1] = thick_prof / max_depth
                params.append({
                    'conductivities': conds.tolist(),
                    'thicknesses': ths,
                    'interface_depths': b.tolist(),
                    'snr': snr, 'n_layers': 3,
                })
        if return_params:
            return data, labels, params
        return data, labels

    def save_data(self, data: np.ndarray, labels: np.ndarray,
                  file_prefix: str):
        np.save(f'{file_prefix}_data.npy', data)
        np.save(f'{file_prefix}_labels.npy', labels)

    def load_data(self, file_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(f'{file_prefix}_data.npy')
        labels = np.load(f'{file_prefix}_labels.npy')
        return data, labels
