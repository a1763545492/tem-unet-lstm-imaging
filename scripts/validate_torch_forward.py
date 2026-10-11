# -*- coding: utf-8 -*-
"""TorchDLFTEMForward 验证脚本
1) 与 SimPEG Simulation1DLayered 的数值一致性（均匀/层状/随机模型）
2) 对 σ 的梯度有限差分检查
运行：python scripts/validate_torch_forward.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.forward.tem_forward_simpeg import TEMForwardSimPEG
from src.forward.tem_forward_torch import TorchDLFTEMForward


def main():
    time_gates = np.logspace(-6, -2, 30)
    dz, nz = 10.0, 30
    thicks = np.full(nz - 1, dz)
    fwd_np = TEMForwardSimPEG(time_gates)
    fwd_t = TorchDLFTEMForward(time_gates, thicks)

    def simpeg_30layer(conds3, boundaries):
        """三层参数 → 30网格电导率 → SimPEG 30层正演"""
        d = np.arange(nz) * dz
        prof = np.where(d < boundaries[0], conds3[0],
                        np.where(d < boundaries[1], conds3[1], conds3[2]))
        # 30 个电导率 + 30 个厚度(末层0)，step_dbdt 自动去掉末层厚度
        r = np.abs(fwd_np.step_dbdt(prof.tolist(),
                                    np.r_[thicks, 0.0].tolist()))
        return r, prof

    # ---- 测试1：均匀半空间（30层同σ）----
    sig = np.full(nz, 0.02)
    ref = np.abs(fwd_np.step_dbdt([0.02], [0.0]))
    out = fwd_t(torch.tensor(sig)[None])[0].numpy()
    err = np.max(np.abs(out - ref) / np.abs(ref))
    print(f'[1] 均匀半空间 σ=0.02: 最大相对误差 = {err:.2e}')

    # ---- 测试2：三层模型 H型 ----
    conds3 = [0.05, 0.005, 0.02]
    b = [60.0, 160.0]
    d = np.arange(nz) * dz
    prof = np.where(d < b[0], conds3[0], np.where(d < b[1], conds3[1], conds3[2]))
    ref3 = np.abs(fwd_np.step_dbdt(prof.tolist(), np.r_[thicks, 0.0].tolist()))
    out3 = fwd_t(torch.tensor(prof)[None])[0].numpy()
    err3 = np.max(np.abs(out3 - ref3) / np.abs(ref3))
    print(f'[2] 三层H型模型: 最大相对误差 = {err3:.2e}')

    # ---- 测试3：10个随机三层模型 ----
    rng = np.random.default_rng(7)
    errs = []
    for _ in range(10):
        c3 = 10 ** rng.uniform(-2.5, -0.7, 3)
        bnds = sorted(rng.uniform(30, 240, 2))
        p = np.where(d < bnds[0], c3[0],
                     np.where(d < bnds[1], c3[1], c3[2]))
        r = np.abs(fwd_np.step_dbdt(p.tolist(), np.r_[thicks, 0.0].tolist()))
        o = fwd_t(torch.tensor(p)[None])[0].numpy()
        errs.append(np.max(np.abs(o - r) / np.abs(r)))
    print(f'[3] 10个随机模型: 最大相对误差 均值={np.mean(errs):.2e} '
          f'最差={np.max(errs):.2e}')

    # ---- 测试4：梯度有限差分检查（对中间时间道）----
    sig_t = torch.tensor(prof, dtype=torch.float64, requires_grad=True)
    out4 = fwd_t(sig_t[None])[0]
    t_idx = 15
    out4[t_idx].backward()
    ad = sig_t.grad.numpy()
    fd = np.zeros(nz)
    eps = 1e-5
    for j in range(nz):
        pp = prof.copy()
        pp[j] += eps
        op = fwd_t(torch.tensor(pp)[None])[0, t_idx].item()
        pp[j] -= 2 * eps
        om = fwd_t(torch.tensor(pp)[None])[0, t_idx].item()
        fd[j] = (op - om) / (2 * eps)
    rel = np.max(np.abs(ad - fd) / (np.abs(fd) + 1e-12))
    # 跳过深度过大、灵敏度近0的节点
    mask = np.abs(fd) > np.max(np.abs(fd)) * 1e-4
    rel_m = np.max(np.abs(ad[mask] - fd[mask]) / np.abs(fd[mask]))
    print(f'[4] 梯度有限差分: 敏感层最大相对误差 = {rel_m:.2e} '
          f'(敏感节点数 {mask.sum()}/{nz})')

    # ---- 测试5：批量 + 计时 ----
    import time
    xb = torch.tensor(np.stack([prof] * 32), requires_grad=True)
    t0 = time.time()
    ob = fwd_t(xb)
    t1 = time.time()
    print(f'[5] batch=32 前向耗时 = {t1-t0:.2f}s, 输出形状 {tuple(ob.shape)}')
    ob.sum().backward()
    print(f'    +反向耗时 = {time.time()-t1:.2f}s')


if __name__ == '__main__':
    main()
