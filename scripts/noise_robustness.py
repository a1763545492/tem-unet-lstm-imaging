# -*- coding: utf-8 -*-
"""抗噪性测试：不同 SNR 下各方法反演精度对比

对同一批真值模型，分别叠加 SNR=5/10/15/20/25/30 dB 噪声，
比较 本文模型(physics_model.pth) 与 OCCAM-1D 的 log10σ RMSE，
重复 N 次取均值±标准差，绘制 RMSE-SNR 曲线（论文标准图）。
运行：python scripts/noise_robustness.py [--repeats 10]
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

from src.forward.tem_forward import TEMForward
from src.models.attention_lstm import UNetAttentionLSTM
from src.baselines import Occam1D
from src.utils.metrics import rmse
from src.visualization.figure_manager import FigureManager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def true_logsigma(conds, ths, max_depth, nz):
    b = np.cumsum(ths)[:2]
    d = np.arange(nz) * (max_depth / nz)
    return np.log10(np.where(d < b[0], conds[0],
                             np.where(d < b[1], conds[1], conds[2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--model', default='models/physics_model.pth')
    ap.add_argument('--occam-layers', type=int, default=12)
    args = ap.parse_args()

    time_gates = np.logspace(-6, -2, 30)
    max_depth, nz = 300.0, 30
    depth_grid = np.arange(nz) * (max_depth / nz)
    fwd = TEMForward(time_gates)
    rng = np.random.default_rng(123)

    model = UNetAttentionLSTM(input_channels=1, input_length=30,
                              num_classes=2)
    state = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()

    snrs = [5, 10, 15, 20, 25, 30]
    res_nn, res_oc = {s: [] for s in snrs}, {s: [] for s in snrs}

    for s in snrs:
        for rep in range(args.repeats):
            conds = 10 ** rng.uniform(-2.5, -0.7, 3)
            ths = [rng.uniform(30, 120), rng.uniform(30, 120), 0.0]
            t_log = true_logsigma(conds, ths, max_depth, nz)
            with np.errstate(all='ignore'):
                r = fwd.layered_earth_response(list(conds), list(ths))
                r = fwd.add_noise(r, snr=float(s))
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

            # 神经网络
            mu, sd = r.mean(), r.std()
            x = torch.from_numpy(((r - mu) / (sd + 1e-12))
                                 [None, None, :].astype(np.float32))
            with torch.no_grad():
                out = model(x)
                out = out[0] if isinstance(out, tuple) else out
            res_nn[s].append(rmse(t_log, out[0, 0].numpy()))

            # OCCAM（输出电导率 → log10）
            oc = Occam1D(time_gates, n_layers=args.occam_layers,
                         max_depth=max_depth, max_iter=4)
            _, c, _ = oc.invert(r)
            c_grid = np.interp(depth_grid, oc.depth_grid, c)
            res_oc[s].append(rmse(t_log, np.log10(np.maximum(c_grid, 1e-8))))
        print(f'SNR={s:>2}dB  NN: {np.mean(res_nn[s]):.4f}±{np.std(res_nn[s]):.4f}'
              f'  OCCAM: {np.mean(res_oc[s]):.4f}±{np.std(res_oc[s]):.4f}')

    fig, ax = plt.subplots(figsize=(7, 5))
    for res, name, color in [(res_nn, '本文模型', 'b'),
                             (res_oc, 'OCCAM-1D', 'r')]:
        m = [np.mean(res[s]) for s in snrs]
        e = [np.std(res[s]) for s in snrs]
        ax.errorbar(snrs, m, yerr=e, fmt=f'{color}-o', capsize=4, label=name)
    ax.set_xlabel('信噪比 SNR / dB')
    ax.set_ylabel('log10(σ) RMSE')
    ax.set_title('抗噪性对比：不同SNR下的反演精度')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/fig_noise_robustness.png', dpi=600,
                bbox_inches='tight')
    FigureManager().register('fig_noise_robustness.png', usage='paper',
                             description='抗噪性：RMSE-SNR曲线(NN vs OCCAM)')
    print('图件已保存: figures/fig_noise_robustness.png')


if __name__ == '__main__':
    main()
