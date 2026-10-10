import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 2):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int = 30, hidden_size: int = 128, 
                 num_layers: int = 2, bidirectional: bool = True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.permute(0, 2, 1)
        out, (h_n, c_n) = self.lstm(x)
        h_n = h_n.view(-1, self.num_directions, self.hidden_size)[-1:]
        c_n = c_n.view(-1, self.num_directions, self.hidden_size)[-1:]
        return out, (h_n, c_n)


class UNetLSTM(nn.Module):
    def __init__(self, input_channels: int = 1, input_length: int = 30, 
                 lstm_hidden_size: int = 128, num_classes: int = 2):
        super(UNetLSTM, self).__init__()
        
        self.unet = UNet1D(n_channels=input_channels, n_classes=lstm_hidden_size)
        self.lstm_encoder = LSTMEncoder(input_size=input_length, hidden_size=lstm_hidden_size)
        
        self.fusion_conv = nn.Conv1d(lstm_hidden_size * 2, lstm_hidden_size, kernel_size=1)
        self.bn = nn.BatchNorm1d(lstm_hidden_size)
        self.relu = nn.ReLU(inplace=True)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(lstm_hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unet_out = self.unet(x)
        
        lstm_out, _ = self.lstm_encoder(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = F.adaptive_avg_pool1d(lstm_out, unet_out.size()[2])
        
        fused = torch.cat([unet_out, lstm_out], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        output = self.decoder(fused)
        return output

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))