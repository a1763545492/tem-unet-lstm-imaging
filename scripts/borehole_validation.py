# -*- coding: utf-8 -*-
"""合成钻孔验证统计：虚拟钻孔揭露信息 vs 反演结果的一致性检验

项目尚处室内研究阶段（无真实钻孔资料），按论文通行做法构造虚拟钻孔：
在随机三层模型上模拟"钻孔揭露的层界面深度与分层电导率"，检验反演
剖面在钻孔位置处能否复现钻孔信息，并同步校验 MC Dropout 不确定性
置信区间的标定（覆盖率-可靠性曲线）。

统计口径（钻孔视角，均为层平均 log10σ）：
  1) 界面深度误差：钻孔揭露界面 vs 反演剖面梯度拾取界面（最近匹配）
  2) 分层电导率偏差：各层反演层平均 log10σ - 钻孔层平均 log10σ（NN vs OCCAM）
  3) 区间标定：名义置信水平 50%~95% 的实测覆盖率（可靠性曲线）

输出：output/borehole_validation.json + figures/fig_borehole_validation.png
运行：python scripts/borehole_validation.py [--holes 100] [--passes 30]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm as _norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.forward.tem_forward_simpeg import TEMForwardSimPEG as TEMForward
from src.models.attention_lstm import UNetAttentionLSTM
from src.baselines import Occam1D
from src.inference import predict_with_uncertainty
from src.utils.metrics import interface_depth_error
from src.visualization.figure_manager import FigureManager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holes', type=int, default=100)
    ap.add_argument('--passes', type=int, default=30,
                    help='MC Dropout 前向次数')
    ap.add_argument('--model', default='models/physics_model.pth')
    args = ap.parse_args()

    time_gates = np.logspace(-6, -2, 30)
    max_depth, nz = 300.0, 30
    depth_grid = np.arange(nz) * (max_depth / nz)
    fwd = TEMForward(time_gates)
    rng = np.random.default_rng(2025)

    model = UNetAttentionLSTM(input_channels=1, input_length=30,
                              num_classes=2)
    state = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()

    with open('models/norm_params.json', 'r', encoding='utf-8') as f:
        prm = json.load(f)
    mu, sd = prm['mu'], prm['sd']

    grid_thicks = np.r_[np.full(nz - 1, max_depth / nz), 0.0].tolist()

    # 批量生成虚拟钻孔（正演与训练协议一致：30 网格阶梯 + SNR 15-30dB）
    R = np.zeros((args.holes, time_gates.size), dtype=np.float32)
    truths, bnds = [], []
    for i in range(args.holes):
        conds = 10 ** rng.uniform(-2.5, -0.7, 3)
        ths = [rng.uniform(30, 120), rng.uniform(30, 120), 0.0]
        b = np.cumsum(ths)[:2]
        d = np.arange(nz) * (max_depth / nz)
        prof = np.where(d < b[0], conds[0],
                        np.where(d < b[1], conds[1], conds[2]))
        with np.errstate(all='ignore'):
            r = fwd.layered_earth_response(prof.tolist(), grid_thicks)
            r = fwd.add_noise(r, snr=float(rng.uniform(15, 30)))
        R[i] = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        truths.append(np.log10(prof))
        bnds.append(b)

    # NN 批量 MC Dropout 推理（全局 log 归一化与训练一致）
    log_R = np.log10(np.maximum(R, 1e-30))
    x = torch.from_numpy(
        ((log_R - mu) / (sd + 1e-12))[:, None, :].astype(np.float32))
    mean, std = predict_with_uncertainty(model, x, n_passes=args.passes)
    nn_log, nn_std = mean[:, 0, :], std[:, 0, :]        # (H, 30)

    # 逐孔统计
    ide_nn, ide_oc = [], []                              # 界面深度误差
    layer_err = {m: {L: [] for L in range(3)}
                 for m in ('NN', 'OCCAM')}                # 分层带符号偏差
    cal_pairs, val_pairs = [], []                        # 标定/验证集 (偏差, 层平均std)
    n_det = {'NN': 0, 'OCCAM': 0}                        # 界面检出数
    half = args.holes // 2                               # 前 1/2 标定、后 1/2 验证
    for i in range(args.holes):
        b, t_log = bnds[i], truths[i]
        layer_idx = np.where(depth_grid < b[0], 0,
                             np.where(depth_grid < b[1], 1, 2))

        e, _ = interface_depth_error(t_log, nn_log[i], depth_grid)
        if not np.isnan(e):
            n_det['NN'] += 1
        ide_nn.append(e)

        oc = Occam1D(time_gates, n_layers=12, max_depth=max_depth,
                     max_iter=4)
        _, c, _ = oc.invert(R[i])
        c_grid = np.interp(depth_grid, oc.depth_grid, c)
        oc_log = np.log10(np.maximum(c_grid, 1e-8))
        e2, _ = interface_depth_error(t_log, oc_log, depth_grid)
        if not np.isnan(e2):
            n_det['OCCAM'] += 1
        ide_oc.append(e2)

        for L in range(3):
            m = layer_idx == L
            t_mean = t_log[m].mean()
            err_nn = float(nn_log[i][m].mean() - t_mean)
            err_oc = float(oc_log[m].mean() - t_mean)
            layer_err['NN'][L].append(err_nn)
            layer_err['OCCAM'][L].append(err_oc)
            pair = (err_nn, float(nn_std[i][m].mean()))
            (cal_pairs if i < half else val_pairs).append(pair)
        if (i + 1) % 10 == 0:
            print(f'  已完成 {i + 1}/{args.holes} 孔')

    # 不确定性标定：MC Dropout 原生逐节点 std 通常低估不确定性
    # （仅认知不确定性，不含测量噪声），用标定集估计线性缩放因子 s，
    # 在独立验证集上检验标定前后的覆盖率-可靠性曲线
    ps = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    cal_errs = np.array([p[0] for p in cal_pairs])
    cal_sds = np.array([p[1] for p in cal_pairs])
    val_errs = np.array([p[0] for p in val_pairs])
    val_sds = np.array([p[1] for p in val_pairs])
    z90 = _norm.ppf(0.95)
    s_factor = float(np.median(np.abs(cal_errs) / (z90 * cal_sds)))
    cov_raw = [float(np.mean(np.abs(val_errs)
                             <= _norm.ppf(0.5 + p / 2) * val_sds))
               for p in ps]
    cov_cal = [float(np.mean(np.abs(val_errs)
                             <= _norm.ppf(0.5 + p / 2) * val_sds * s_factor))
               for p in ps]

    ide_nn = np.asarray(ide_nn, float)
    ide_oc = np.asarray(ide_oc, float)
    summary = {
        'n_holes': args.holes, 'mc_passes': args.passes,
        'NN': {
            '界面深度误差均值m': float(np.nanmean(ide_nn)),
            '界面深度误差std': float(np.nanstd(ide_nn)),
            '界面检出率': n_det['NN'] / args.holes,
            '分层平均偏差log10σ': {f'第{L + 1}层': {
                'mean': float(np.mean(layer_err['NN'][L])),
                'std': float(np.std(layer_err['NN'][L]))} for L in range(3)},
        },
        'OCCAM': {
            '界面深度误差均值m': float(np.nanmean(ide_oc)),
            '界面深度误差std': float(np.nanstd(ide_oc)),
            '界面检出率': n_det['OCCAM'] / args.holes,
            '分层平均偏差log10σ': {f'第{L + 1}层': {
                'mean': float(np.mean(layer_err['OCCAM'][L])),
                'std': float(np.std(layer_err['OCCAM'][L]))} for L in range(3)},
        },
        '不确定性标定': {
            '标定孔数': half, '验证孔数': args.holes - half,
            '标定因子': s_factor,
            '覆盖率90%实测(未标定)': cov_raw[4],
            '覆盖率90%实测(标定后)': cov_cal[4],
            '可靠性曲线': {'levels': ps.tolist(),
                           'coverage_raw': cov_raw,
                           'coverage_calibrated': cov_cal},
        },
    }
    os.makedirs('output', exist_ok=True)
    with open('output/borehole_validation.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 三联图：(a)界面误差分布 (b)分层一致性箱线 (c)可靠性曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    bins = np.arange(0, 85, 8)
    axes[0].hist(ide_nn[~np.isnan(ide_nn)], bins=bins, alpha=0.65,
                 color='b', label='本文模型', density=True)
    axes[0].hist(ide_oc[~np.isnan(ide_oc)], bins=bins, alpha=0.55,
                 color='r', label='OCCAM-1D', density=True)
    axes[0].set_xlabel('界面深度误差 / m')
    axes[0].set_ylabel('概率密度')
    axes[0].set_title('(a) 钻孔揭露界面深度误差分布')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    data_nn = [layer_err['NN'][L] for L in range(3)]
    data_oc = [layer_err['OCCAM'][L] for L in range(3)]
    axes[1].boxplot(data_nn, positions=np.arange(3) - 0.18, widths=0.32,
                    patch_artist=True, boxprops=dict(facecolor='#8ab4f8'),
                    medianprops=dict(color='b'))
    axes[1].boxplot(data_oc, positions=np.arange(3) + 0.18, widths=0.32,
                    patch_artist=True, boxprops=dict(facecolor='#f6a5a5'),
                    medianprops=dict(color='r'))
    axes[1].axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(['第1层', '第2层', '第3层'])
    axes[1].set_ylabel('层平均 log10σ 偏差')
    axes[1].set_title('(b) 分层电导率一致性（蓝=本文，红=OCCAM）')
    axes[1].grid(alpha=0.3, axis='y')

    axes[2].plot([0.4, 1.0], [0.4, 1.0], 'k--', lw=1, alpha=0.6,
                 label='理想标定')
    axes[2].plot(ps, cov_raw, 'r-s', label='MC Dropout 原生')
    axes[2].plot(ps, cov_cal, 'b-o',
                 label=f'方差标定后 (s={s_factor:.2f})')
    axes[2].set_xlabel('名义置信水平')
    axes[2].set_ylabel('实测覆盖率')
    axes[2].set_title('(c) 置信区间标定（层平均 log10σ）')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig('figures/fig_borehole_validation.png', dpi=600,
                bbox_inches='tight')
    FigureManager().register('fig_borehole_validation.png', usage='paper',
                             description='合成钻孔验证：界面误差/分层一致性/不确定性标定')

    print('\n===== 合成钻孔验证统计 =====')
    print(f"钻孔数={args.holes}, MC前向={args.passes}")
    print(f"界面深度误差  NN: {summary['NN']['界面深度误差均值m']:.1f}±"
          f"{summary['NN']['界面深度误差std']:.1f} m "
          f"(检出率{summary['NN']['界面检出率']:.0%})  "
          f"OCCAM: {summary['OCCAM']['界面深度误差均值m']:.1f}±"
          f"{summary['OCCAM']['界面深度误差std']:.1f} m "
          f"(检出率{summary['OCCAM']['界面检出率']:.0%})")
    for L in range(3):
        nn_s = summary['NN']['分层平均偏差log10σ'][f'第{L + 1}层']
        oc_s = summary['OCCAM']['分层平均偏差log10σ'][f'第{L + 1}层']
        print(f"第{L + 1}层偏差  NN: {nn_s['mean']:+.3f}±{nn_s['std']:.3f}"
              f"  OCCAM: {oc_s['mean']:+.3f}±{oc_s['std']:.3f}")
    print(f"90%置信区间实测覆盖率: 未标定 {cov_raw[4]:.1%} → "
          f"标定后 {cov_cal[4]:.1%}（理想 90%，标定因子 s={s_factor:.2f}，"
          f"标定/验证孔数 {half}/{args.holes - half}）")
    print('图件已保存: figures/fig_borehole_validation.png')
    print('统计已保存: output/borehole_validation.json')


if __name__ == '__main__':
    main()
