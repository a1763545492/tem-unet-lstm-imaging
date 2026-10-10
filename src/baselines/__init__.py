# -*- coding: utf-8 -*-
"""传统基线方法模块（论文对比实验用）

- BostickInverter: Bostick 变换近似反演
- Occam1D: 光滑约束最小构造 1D 反演（Gauss-Newton + 有限差分 Jacobian）
- UNet1D / ResNet1D: 消融对比网络
"""
from .bostick import BostickInverter
from .occam1d import Occam1D
from .networks import UNet1D, ResNet1D

__all__ = ['BostickInverter', 'Occam1D', 'UNet1D', 'ResNet1D']
