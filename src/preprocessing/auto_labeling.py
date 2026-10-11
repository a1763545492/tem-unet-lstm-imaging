# -*- coding: utf-8 -*-
"""自动标注管线：合成数据-标签对生成（新协议）

2026-10-11 协议统一：本模块旧版采用 1-5 层零填充标签 (2, max_layers)、
原始 σ 标签、独立采样协议，与主训练管线（scripts/train_physics.py，
30 网格阶梯剖面、log10σ 标签、(2, nz) 节点标签）物理不一致，
且标签维度与主网络输出 (B, 2, 30) 不匹配，无法直接训练。

现统一委托 TEMForward.generate_synthetic_data(return_params=True)
作为唯一协议源（30 网格阶梯正演 + log10σ/厚度节点标签 + 种子可复现），
本模块保留：层参数→标签转换器（generate_labels_from_params）、
地形变体（generate_with_terrain）、数据集存取与训练管线入口。
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from src.forward.tem_forward import TEMForward


class AutoLabelingPipeline:
    """合成数据自动标注（协议与 train_physics.make_dataset 严格一致）。"""

    def __init__(self, forward_model: TEMForward = None,
                 max_depth: float = 300.0, nz: int = 30,
                 snr_range: Tuple[float, float] = (15.0, 30.0),
                 sigma_range: Tuple[float, float] = (-2.5, -0.7),
                 interface_range: Tuple[float, float] = (30.0, 120.0)):
        self.forward_model = (TEMForward() if forward_model is None
                              else forward_model)
        self.max_depth = float(max_depth)
        self.nz = int(nz)
        self.snr_range = tuple(snr_range)
        self.sigma_range = tuple(sigma_range)
        self.interface_range = tuple(interface_range)

    # ---------- 核心生成（委托严格协议源） ----------
    def generate_data_label_pair(self, n_samples: int = 1000, seed: int = 42
                                 ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """生成 (响应, 标签, 层参数) 三元组。

        Returns:
            data:   (n, T) 含噪 dB/dt（物理幅值，未归一化）
            labels: (n, 2, nz) [log10σ; 厚度/max_depth] 节点剖面
            params: 逐样本层参数字典（含 snr / interface_depths）
        """
        data, labels, params = self.forward_model.generate_synthetic_data(
            n_samples=n_samples, max_depth=self.max_depth, nz=self.nz,
            seed=seed, snr_range=self.snr_range,
            sigma_range=self.sigma_range,
            interface_range=self.interface_range,
            return_params=True)
        return data, labels, params

    # ---------- 层参数 → 节点标签转换 ----------
    def generate_labels_from_params(self, params: List[Dict]) -> np.ndarray:
        """层参数字典列表 → (n, 2, nz) 节点标签（新协议语义）。

        每个样本的层参数（conductivities/thicknesses，末层为半空间）被
        离散化为 nz 节点阶梯剖面：ch0=log10σ，ch1=节点所在层厚度/max_depth
        （半空间节点厚度记 0）。超出 max_depth 的界面按深度截断。
        """
        nz, dz = self.nz, self.max_depth / self.nz
        d = np.arange(nz) * dz
        labels = np.zeros((len(params), 2, nz), dtype=np.float64)
        for i, param in enumerate(params):
            conds = np.asarray(param.get('conductivities', [1.0]), float)
            thicks = np.asarray(param.get('thicknesses',
                                          [self.max_depth]), float)
            if len(thicks) == len(conds):
                thicks = thicks[:-1]          # 半空间厚度不入界面累计
            b = np.cumsum(thicks)             # 界面深度
            thick_layers = np.r_[thicks, 0.0]  # 逐层厚度（半空间记 0）
            idx = np.clip(np.searchsorted(b, d, side='right'),
                          0, min(len(thick_layers), len(conds)) - 1)
            labels[i, 0] = np.log10(conds[idx])
            labels[i, 1] = thick_layers[idx] / self.max_depth
        return labels

    # ---------- 地形变体 ----------
    def generate_with_terrain(self, terrain_model, n_samples: int = 1000,
                              seed: int = 42
                              ) -> Tuple[list, np.ndarray, List[Dict]]:
        """地形影响变体：同一随机模型，多测点地形响应 + 新协议标签。

        Returns:
            data:    样本列表（各样本 (n_stations, T)，测点数可变）
            labels:  (n, 2, nz) 节点标签（与平坦协议一致）
            params:  层参数 + 测点坐标
        """
        rng = np.random.default_rng(seed)
        dz = self.max_depth / self.nz
        grid_thicks = np.r_[np.full(self.nz - 1, dz), 0.0].tolist()
        d = np.arange(self.nz) * dz
        data, labels, params = [], [], []
        for _ in range(n_samples):
            conds = 10 ** rng.uniform(*self.sigma_range, 3)
            ths = [rng.uniform(*self.interface_range),
                   rng.uniform(*self.interface_range), 0.0]
            b = np.cumsum(ths)[:2]
            cond_prof = np.where(d < b[0], conds[0],
                                 np.where(d < b[1], conds[1], conds[2]))
            thick_prof = np.where(d < b[0], ths[0],
                                  np.where(d < b[1], ths[1], 0.0))
            n_stations = int(rng.integers(5, 20))
            stations = rng.uniform(-100, 100, (n_stations, 2))
            responses = terrain_model.calculate_terrain_response(
                stations, cond_prof.tolist(), grid_thicks, self.forward_model)
            snr = float(rng.uniform(*self.snr_range))
            noisy = [self.forward_model.add_noise(r, snr=snr)
                     for r in responses]
            data.append(np.asarray(noisy))
            labels.append(np.stack([np.log10(cond_prof),
                                    thick_prof / self.max_depth]))
            params.append({'conductivities': conds.tolist(),
                           'thicknesses': ths,
                           'interface_depths': b.tolist(),
                           'snr': snr, 'n_layers': 3,
                           'n_stations': n_stations,
                           'stations': stations.tolist()})
        labels = np.asarray(labels)
        return data, labels, params

    # ---------- 数据集存取 ----------
    def save_dataset(self, data, labels: np.ndarray, params: List[Dict],
                     save_dir: str = './data', dataset_name: str = 'dataset'):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'{dataset_name}_data.npy'),
                np.asarray(data))
        np.save(os.path.join(save_dir, f'{dataset_name}_labels.npy'), labels)
        np.save(os.path.join(save_dir, f'{dataset_name}_params.npy'),
                np.array(params, dtype=object))
        print(f"Dataset saved to {save_dir}")
        print(f"Data shape: {np.asarray(data).shape}")
        print(f"Labels shape: {labels.shape}")

    def load_dataset(self, save_dir: str = './data',
                     dataset_name: str = 'dataset'
                     ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        data = np.load(os.path.join(save_dir, f'{dataset_name}_data.npy'))
        labels = np.load(os.path.join(save_dir,
                                      f'{dataset_name}_labels.npy'))
        params = np.load(os.path.join(save_dir,
                                      f'{dataset_name}_params.npy'),
                         allow_pickle=True).tolist()
        return data, labels, params

    # ---------- 批量 / 训练管线入口 ----------
    def generate_batch(self, batch_size: int = 32, seed: int = 42
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """新协议批量生成。旧 max_layers 参数已随协议废除。"""
        data, labels, _ = self.generate_data_label_pair(batch_size, seed=seed)
        return data, labels

    def create_training_pipeline(self, config: Dict
                                 ) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        """生成训练/验证集（全局 log 归一化在训练脚本侧完成）。"""
        n_train = config.get('n_train', 10000)
        n_val = config.get('n_val', 2000)
        seed = config.get('seed', 42)
        train_data, train_labels, _ = self.generate_data_label_pair(
            n_train, seed=seed)
        val_data, val_labels, _ = self.generate_data_label_pair(
            n_val, seed=seed + 1)
        return train_data, train_labels, val_data, val_labels
