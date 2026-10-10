# -*- coding: utf-8 -*-
"""创新点B：MC Dropout 不确定性量化

推理阶段保持 Dropout 开启，对同一输入做 T 次随机前向传播，
输出均值（最优估计）与逐节点标准差（置信度）。
标准差大的区域（通常为异常体边界）表示反演结果可信度低。

参考：Gal & Ghahramani (2016) Dropout as a Bayesian Approximation。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def enable_dropout(model: nn.Module):
    """仅将 Dropout 层置为训练模式（BatchNorm 等保持 eval）"""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model: nn.Module, x: torch.Tensor,
                             n_passes: int = 50,
                             device: str = 'cpu'
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """MC Dropout 推理。

    Args:
        model: 含 Dropout 层的网络
        x: (B, 1, T) 输入
        n_passes: 随机前向次数（论文建议 30~100）

    Returns:
        (mean, std): 各 (B, C, L)，预测均值与逐节点标准差
    """
    model.eval()
    enable_dropout(model)
    x = x.to(device)
    preds = []
    for _ in range(n_passes):
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        preds.append(out.cpu().numpy())
    model.eval()
    P = np.stack(preds)          # (T, B, C, L)
    return P.mean(axis=0), P.std(axis=0)
