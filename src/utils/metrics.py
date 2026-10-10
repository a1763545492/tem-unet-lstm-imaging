# -*- coding: utf-8 -*-
"""反演精度定量评估指标集（论文对比实验用）

包含：RMSE / MAE / R² / 界面深度误差 / 异常体定位偏差 / SSIM / 相对误差
所有指标输入为一维电导率(或电阻率)深度剖面或二维断面，与具体模型无关。
"""
import numpy as np
from typing import Dict, Optional, Tuple


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray,
                   eps: float = 1e-8) -> np.ndarray:
    """逐点相对误差 |y_pred - y_true| / (|y_true| + eps)"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return np.abs(y_pred - y_true) / (np.abs(y_true) + eps)


def _find_boundaries(profile: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """从剖面梯度中拾取层界面位置（梯度峰值索引）"""
    grad = np.abs(np.diff(profile))
    if grad.max() < 1e-12:
        return np.array([], dtype=int)
    thresh = ratio * grad.max()
    idx = np.where(grad >= thresh)[0]
    # 合并相邻索引，取各连续段最大梯度位置
    if len(idx) == 0:
        return np.array([], dtype=int)
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return np.array([g[np.argmax(grad[g])] for g in groups], dtype=int)


def interface_depth_error(true_profile: np.ndarray, pred_profile: np.ndarray,
                          depth_grid: np.ndarray,
                          ratio: float = 0.5) -> Tuple[float, float]:
    """界面深度误差：真值界面与预测界面的最近匹配深度差。

    Returns:
        (平均绝对误差 m, 相对误差 %，占最大探测深度比例)
        无界面可匹配时返回 (nan, nan)。
    """
    tb = _find_boundaries(np.asarray(true_profile, float), ratio)
    pb = _find_boundaries(np.asarray(pred_profile, float), ratio)
    if len(tb) == 0 or len(pb) == 0:
        return float('nan'), float('nan')
    errs = [np.min(np.abs(depth_grid[pb] - depth_grid[t])) for t in tb]
    mean_err = float(np.mean(errs))
    max_depth = float(depth_grid[-1] - depth_grid[0])
    return mean_err, 100.0 * mean_err / (max_depth + 1e-12)


def anomaly_center_error(true_profile: np.ndarray, pred_profile: np.ndarray,
                         depth_grid: np.ndarray) -> float:
    """异常体（高导/低阻区）中心深度定位偏差（m）"""
    def _centroid(p):
        w = np.maximum(p - np.median(p), 0.0)
        return float(np.sum(depth_grid * w) / (np.sum(w) + 1e-12)) if w.sum() > 0 \
            else float('nan')
    return abs(_centroid(np.asarray(true_profile, float))
               - _centroid(np.asarray(pred_profile, float)))


def _gaussian_kernel1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def ssim(img1: np.ndarray, img2: np.ndarray,
         data_range: Optional[float] = None) -> float:
    """结构相似度（1D剖面或2D断面通用，高斯窗口 11，sigma 1.5）"""
    x, y = np.asarray(img1, float), np.asarray(img2, float)
    if x.shape != y.shape:
        raise ValueError('SSIM 输入形状必须一致')
    if data_range is None:
        data_range = max(x.max(), y.max()) - min(x.min(), y.min()) + 1e-12
    k1 = _gaussian_kernel1d()

    def _filt(a):
        if a.ndim == 1:
            return np.convolve(a, k1, mode='same')
        from scipy.ndimage import convolve1d
        a = convolve1d(a, k1, axis=0, mode='nearest')
        return convolve1d(a, k1, axis=1, mode='nearest')

    def _local_var(a, mu):
        return _filt(a * a) - mu * mu

    mu_x, mu_y = _filt(x), _filt(y)
    vx, vy = _local_var(x, mu_x), _local_var(y, mu_y)
    cxy = _filt(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cxy + c2)) / \
        ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2))
    return float(np.mean(s))


def evaluate_profile(true_profile: np.ndarray, pred_profile: np.ndarray,
                     depth_grid: np.ndarray) -> Dict[str, float]:
    """单点一维剖面全套指标（论文表格行）"""
    ide, ide_pct = interface_depth_error(true_profile, pred_profile, depth_grid)
    return {
        'RMSE': rmse(true_profile, pred_profile),
        'MAE': mae(true_profile, pred_profile),
        'R2': r2_score(true_profile, pred_profile),
        '界面深度误差(m)': ide,
        '界面深度误差(%)': ide_pct,
        '异常定位偏差(m)': anomaly_center_error(true_profile, pred_profile,
                                                depth_grid),
        'SSIM': ssim(true_profile, pred_profile),
    }


def evaluate_section(true_section: np.ndarray, pred_section: np.ndarray,
                     depth_grid: np.ndarray) -> Dict[str, float]:
    """二维断面指标：剖面指标取均值 + 断面 SSIM"""
    rows = [evaluate_profile(t, p, depth_grid)
            for t, p in zip(true_section, pred_section)]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out['断面SSIM'] = ssim(true_section, pred_section)
    return out
