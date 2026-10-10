# -*- coding: utf-8 -*-
"""消融对比网络：纯 U-Net1D 与 ResNet1D

输出格式与 UNetAttentionLSTM 一致：(B, num_classes, L)，
便于在同一训练协议下做消融对比。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet1D(nn.Module):
    """纯 U-Net1D（无 LSTM、无注意力）——消融基线"""

    def __init__(self, input_channels: int = 1, input_length: int = 30,
                 num_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, 3, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 移除第二次池化：30个时间道太短，两次池化(30->15->7)信息损失过大
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 2, stride=2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(self.encoder(x))
        diff = x.shape[2] - out.shape[2]
        if diff > 0:
            out = F.pad(out, [diff // 2, diff - diff // 2])
        return out


class ResBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class ResNet1D(nn.Module):
    """ResNet1D 回归网络——主流深度学习基线"""

    def __init__(self, input_channels: int = 1, input_length: int = 30,
                 num_classes: int = 2, n_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(64) for _ in range(n_blocks)])
        self.head = nn.Conv1d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
