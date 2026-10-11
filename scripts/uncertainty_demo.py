# -*- coding: utf-8 -*-
"""不确定性量化演示：生成论文图件（均值反演断面 + 标准差置信图）

加载 models/physics_model.pth，对含低阻异常体的演示剖面做 MC Dropout 推理，
输出双联图：上=均值电阻率断面，下=逐节点标准差（置信度）图。
运行：python scripts/uncertainty_demo.py [--passes 50]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.forward.tem_forward_simpeg import TEMForwardSimPEG as TEMForward
from src.models.attention_lstm import UNetAttentionLSTM
from src.inference import predict_with_uncertainty
from src.visualization.figure_manager import FigureManager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def make_profile(time_gates, n_stations=50, seed=7):
    """含采空区低阻（高导）异常体的演示剖面

    与训练协议严格一致：30 网格阶梯剖面正演 + 全局 log 归一化
    （models/norm_params.json），否则模型输入分布漂移、不确定性失真。
    """
    import json
    with open('models/norm_params.json', 'r', encoding='utf-8') as f:
        prm = json.load(f)
    mu_g, sd_g = prm['mu'], prm['sd']
    max_depth, nz = 300.0, 30
    rng = np.random.default_rng(seed)
    fwd = TEMForward(time_gates)
    grid_thicks = np.r_[np.full(nz - 1, max_depth / nz), 0.0].tolist()
    X = []
    for i in range(n_stations):
        if 400 <= i * 20.0 <= 600:
            conds = [0.05, 0.02, 0.05]   # 浅部高导异常（早期道高信噪、可分辨）
        else:
            conds = [0.01, 0.02, 0.05]
        ths = [80, 60, 0.0]
        b = np.cumsum(ths)[:2]
        d = np.arange(nz) * (max_depth / nz)
        prof = np.where(d < b[0], conds[0],
                        np.where(d < b[1], conds[1], conds[2]))
        with np.errstate(all='ignore'):
            r = fwd.layered_earth_response(prof.tolist(), grid_thicks)
            r = fwd.add_noise(r, snr=float(rng.uniform(18, 25)))
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        X.append((np.log10(np.maximum(r, 1e-30)) - mu_g) / (sd_g + 1e-12))
    return np.array(X, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passes', type=int, default=50)
    ap.add_argument('--model', default='models/physics_model.pth')
    args = ap.parse_args()

    time_gates = np.logspace(-6, -2, 30)
    max_depth, nz = 300.0, 30

    X = make_profile(time_gates)
    model = UNetAttentionLSTM(input_channels=1, input_length=30,
                              num_classes=2)
    state = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state.get('model_state_dict', state))
    print(f'模型已加载: {args.model}，MC 推理 {args.passes} 次...')

    x = torch.from_numpy(X).unsqueeze(1)
    mean, std = predict_with_uncertainty(model, x, n_passes=args.passes)
    rho_mean = 10.0 ** (-mean[:, 0, :])      # log10σ → ρ
    rho_std = mean[:, 0, :]                   # 对数空间标准差
    unc = std[:, 0, :]

    xs = np.arange(X.shape[0]) * 20.0
    zs = np.arange(nz) * (max_depth / nz)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    im0 = axes[0].pcolormesh(xs, zs, rho_mean.T, cmap='jet', shading='auto')
    axes[0].invert_yaxis()
    axes[0].set_ylabel('深度 / m')
    axes[0].set_title('MC Dropout 均值反演断面（T={}次）'.format(args.passes))
    fig.colorbar(im0, ax=axes[0], label='电阻率 / (Ω·m)')

    im1 = axes[1].pcolormesh(xs, zs, unc.T, cmap='hot', shading='auto')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('距离 / m')
    axes[1].set_ylabel('深度 / m')
    axes[1].set_title('反演不确定性（log10σ 标准差，越大可信度越低）')
    fig.colorbar(im1, ax=axes[1], label='标准差')
    fig.tight_layout()

    fm = FigureManager()
    raw_path = os.path.join('figures', 'fig_uncertainty.png')
    fig.savefig(raw_path, dpi=600, bbox_inches='tight')
    out = fm.register('fig_uncertainty.png', usage='paper',
                      description='MC Dropout不确定性量化：均值断面与置信度图')
    print(f'图件已保存并注册: {out}')
    print(f'异常区(400-600m)平均不确定性: {unc[20:31].mean():.4f} | '
          f'背景区: {np.delete(unc, np.arange(20, 31)).mean():.4f}')


if __name__ == '__main__':
    main()
