# -*- coding: utf-8 -*-
"""创新点A验证实验：物理约束损失 vs 纯MSE损失 对照训练

同一数据集、同一网络（UNetAttentionLSTM）、同一训练轮数下：
  A组: 纯 MSE（数据驱动基线）
  B组: PhysicsInformedLoss（λ_data·MSE + λ_physics·正演一致性 + λ_smooth·平滑）
输出：
  - models/plain_mse_model.pth / models/physics_model.pth
  - 控制台对比表（验证集电导率 RMSE / R2 / SSIM / 界面深度误差）
运行：python scripts/train_physics.py [--samples 600] [--epochs 20]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.forward.tem_forward import TEMForward
from src.models.attention_lstm import UNetAttentionLSTM
from src.training.physics_loss import (DifferentiableTEMForward,
                                       PhysicsInformedLoss)
from src.utils.metrics import evaluate_profile


def make_dataset(n_samples: int, time_gates: np.ndarray,
                 max_depth: float = 300.0, nz: int = 30, seed: int = 42):
    """随机三层模型数据集，返回 X(归一化), Y, mu, sd"""
    rng = np.random.default_rng(seed)
    fwd = TEMForward(time_gates)
    X = np.zeros((n_samples, time_gates.size), dtype=np.float32)
    Y = np.zeros((n_samples, 2, nz), dtype=np.float32)
    MU = np.zeros(n_samples, dtype=np.float32)
    SD = np.zeros(n_samples, dtype=np.float32)
    with np.errstate(all='ignore'):
        for i in range(n_samples):
            conds = 10 ** rng.uniform(-2.5, -0.7, 3)
            ths = [rng.uniform(30, 120), rng.uniform(30, 120), 0.0]
            resp = fwd.layered_earth_response(list(conds), list(ths))
            resp = fwd.add_noise(resp, snr=float(rng.uniform(15, 30)))
            resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
            mu, sd = resp.mean(), resp.std()
            X[i] = (resp - mu) / (sd + 1e-12)
            MU[i], SD[i] = mu, sd
            b = np.cumsum(ths)[:2]
            d = np.arange(nz) * (max_depth / nz)
            cond_prof = np.where(d < b[0], conds[0],
                                 np.where(d < b[1], conds[1], conds[2]))
            Y[i, 0] = np.log10(cond_prof)  # 对数电导率标签（量纲均衡）
            Y[i, 1] = ths[0] / max_depth
    return X, Y, MU, SD


def train_one(model, loaders, epochs, loss_fn, lr, use_physics: bool,
              device='cpu'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = loaders
    best = np.inf
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for x, y, mu, sd in train_loader:
            x, y, mu, sd = (t.to(device) for t in (x, y, mu, sd))
            opt.zero_grad()
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            if use_physics:
                loss, parts = loss_fn(pred, y, x, (mu, sd))
            else:
                loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            tr += float(loss.detach()) * len(x)
        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, y, mu, sd in val_loader:
                x, y, mu, sd = (t.to(device) for t in (x, y, mu, sd))
                out = model(x)
                pred = out[0] if isinstance(out, tuple) else out
                va += float(F.mse_loss(pred, y)) * len(x)
        n_tr = len(train_loader.dataset)
        n_va = len(val_loader.dataset)
        print(f'  epoch {ep:3d}: train={tr/n_tr:.6f}  val_mse={va/n_va:.6f}')
        best = min(best, va / n_va)
    return model, best


@torch.no_grad()
def evaluate(model, dataset, depth_grid, device='cpu', batch=64):
    model.eval()
    preds = []
    for i in range(0, len(dataset), batch):
        x = dataset.tensors[0][i:i + batch].to(device)
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        preds.append(out[:, 0, :].cpu().numpy())
    # 在对数电导率空间评估（EM反演论文标准：log10σ 的 RMSE/R²/SSIM；
    # 界面误差与定位偏差在 log 空间同样以梯度跳变识别，结论一致）
    P = np.concatenate(preds)
    T = dataset.tensors[1][:, 0, :].numpy()
    rows = [evaluate_profile(t, p, depth_grid) for t, p in zip(T, P)]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=600)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda-physics', type=float, default=0.3)
    ap.add_argument('--lambda-smooth', type=float, default=0.05)
    args = ap.parse_args()

    time_gates = np.logspace(-6, -2, 30)
    max_depth, nz = 300.0, 30
    depth_grid = np.arange(nz) * (max_depth / nz)

    print('生成数据集...')
    X, Y, MU, SD = make_dataset(args.samples, time_gates, max_depth, nz)
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1),
                       torch.from_numpy(Y),
                       torch.from_numpy(MU), torch.from_numpy(SD))
    n_val = int(0.2 * args.samples)
    train_set, val_set = random_split(ds, [args.samples - n_val, n_val],
                                      generator=torch.Generator().manual_seed(0))
    loaders = (DataLoader(train_set, batch_size=args.batch, shuffle=True),
               DataLoader(val_set, batch_size=args.batch))
    val_tensors = TensorDataset(*[t for t in ds[val_set.indices]])

    fwd_op = DifferentiableTEMForward(time_gates, max_depth / nz)
    pi_loss = PhysicsInformedLoss(fwd_op, lambda_physics=args.lambda_physics,
                                  lambda_smooth=args.lambda_smooth)

    results = {}
    os.makedirs('models', exist_ok=True)
    for tag, use_physics, path in [('A组-纯MSE', False,
                                    'models/plain_mse_model.pth'),
                                   ('B组-物理约束', True,
                                    'models/physics_model.pth')]:
        print(f'\n=== {tag} ===')
        torch.manual_seed(0)
        model = UNetAttentionLSTM(input_channels=1, input_length=30,
                                  num_classes=2)
        model, best = train_one(model, loaders, args.epochs, pi_loss,
                                args.lr, use_physics)
        torch.save(model.state_dict(), path)
        metrics = evaluate(model, val_tensors, depth_grid)
        results[tag] = metrics
        print(f'  已保存 {path}')

    print('\n===== 验证集定量对比 =====')
    keys = ['RMSE', 'MAE', 'R2', '界面深度误差(m)', '异常定位偏差(m)', 'SSIM']
    header = f'{"指标":<16}' + ''.join(f'{t:>16}' for t in results)
    print(header)
    for k in keys:
        row = f'{k:<16}'
        for t in results:
            row += f'{results[t][k]:>16.4f}'
        print(row)


if __name__ == '__main__':
    main()
