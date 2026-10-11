# -*- coding: utf-8 -*-
"""重绘 fig8（效率对比）：原 fig8_efficiency_comparison.png 中的耗时为早期
占位示意数据（如 OCCAM 45 s），与实测严重不符。本脚本按 2026-10-11 实测
计时数据重绘，生成版本化新文件（原核心图件保留不动）。

实测环境：CPU 4 线程，单测点，20 次平均（MC 为 T 次前向总耗时）。
运行：python scripts/redraw_fig8.py
"""
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization.figure_manager import FigureManager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 实测单测点耗时 / ms（20 次平均，CPU 4 线程）
timings = {
    'Bostick 换算': 0.24,
    '本文模型\n单次前向': 1.93,
    'OCCAM-1D\n(12层, iter=4)': 10.0,
    '本文模型\nMC Dropout T=50': 100.2,
}
mc_T = np.array([10, 20, 30, 50])
mc_ms = np.array([20.2, 40.0, 59.9, 100.2])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 方法耗时对比（对数轴）
names = list(timings.keys())
vals = np.array(list(timings.values()))
colors = ['#7fb3d5', '#2e86c1', '#f1948a', '#c0392b']
bars = axes[0].bar(names, vals, color=colors, width=0.62)
axes[0].set_yscale('log')
axes[0].set_ylim(0.1, 300)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width() / 2, v * 1.15, f'{v:g} ms',
                 ha='center', fontsize=9)
axes[0].set_ylabel('单测点耗时 / ms（对数轴）')
axes[0].set_title('(a) 各方法单测点推理耗时（实测，CPU 4线程）')
axes[0].grid(axis='y', alpha=0.3, which='both')
axes[0].tick_params(axis='x', labelsize=9)

# (b) MC Dropout 开销随前向次数 T 的变化
axes[1].plot(mc_T, mc_ms, 'o-', color='#2e86c1', label='实测')
axes[1].plot(mc_T, mc_T * 2.0, 'k--', lw=1, alpha=0.6,
             label='线性参考（≈2 ms/次）')
axes[1].set_xlabel('MC Dropout 前向次数 T')
axes[1].set_ylabel('单测点耗时 / ms')
axes[1].set_title('(b) 不确定性量化开销随 T 线性增长')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xticks(mc_T)

fig.tight_layout()
ts = datetime.now().strftime('%Y%m%d%H%M%S')
fname = f'fig8_efficiency_comparison_v{ts}.png'
fig.savefig(f'figures/{fname}', dpi=600, bbox_inches='tight')
FigureManager().register(fname, usage='paper',
                         description='fig8 重绘（实测计时）：方法耗时对比+MC开销线性标度')
print(f'图件已保存: figures/{fname}')
