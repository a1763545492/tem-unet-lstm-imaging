# -*- coding: utf-8 -*-
"""消融实验：同一数据集/同一训练协议下对比 4 种网络结构

对比对象：
  1) UNet1D（纯U-Net，去LSTM与注意力）
  2) ResNet1D（主流DL基线）
  3) UNetAttentionLSTM（本文完整模型）
标签为 log10σ + 归一化层厚；评估用 log10σ 空间全套指标。
结果输出 JSON 并打印论文对比表。
运行：python scripts/ablation_study.py [--samples 800] [--epochs 25]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.baselines import UNet1D, ResNet1D
from src.models.attention_lstm import UNetAttentionLSTM
from src.utils.metrics import evaluate_profile
from train_physics import make_dataset  # 复用同一数据生成协议


def train_model(model, train_loader, val_loader, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, _, _ in train_loader:
            opt.zero_grad()
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, tensors, depth_grid):
    model.eval()
    out = model(tensors[0])
    out = out[0] if isinstance(out, tuple) else out
    P = out[:, 0, :].numpy()
    T = tensors[1][:, 0, :].numpy()
    rows = [evaluate_profile(t, p, depth_grid) for t, p in zip(T, P)]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=800)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    time_gates = np.logspace(-6, -2, 30)
    max_depth, nz = 300.0, 30
    depth_grid = np.arange(nz) * (max_depth / nz)

    print('生成数据集（与物理约束实验同一协议）...')
    X, Y, MU, SD = make_dataset(args.samples, time_gates, max_depth, nz)
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1),
                       torch.from_numpy(Y),
                       torch.from_numpy(MU), torch.from_numpy(SD))
    n_val = int(0.2 * args.samples)
    g = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(ds, [args.samples - n_val, n_val],
                                      generator=g)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)
    val_tensors = [t for t in ds[val_set.indices]]

    nets = {
        'UNet1D(去LSTM)': UNet1D,
        'ResNet1D': ResNet1D,
        'UNet+Attn+LSTM(本文)': UNetAttentionLSTM,
    }
    results = {}
    for name, cls in nets.items():
        print(f'\n=== 训练 {name} ===')
        torch.manual_seed(0)
        model = cls(input_channels=1, input_length=30, num_classes=2)
        model = train_model(model, train_loader, val_loader,
                            args.epochs, args.lr)
        results[name] = evaluate(model, val_tensors, depth_grid)
        print('  完成')

    keys = ['RMSE', 'MAE', 'R2', '界面深度误差(m)', '异常定位偏差(m)', 'SSIM']
    print('\n===== 消融实验对比表（log10σ 空间）=====')
    print(f'{"指标":<16}' + ''.join(f'{n:>22}' for n in results))
    for k in keys:
        print(f'{k:<16}' + ''.join(f'{results[n][k]:>22.4f}' for n in results))

    os.makedirs('output', exist_ok=True)
    with open('output/ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n结果已保存: output/ablation_results.json')


if __name__ == '__main__':
    main()
