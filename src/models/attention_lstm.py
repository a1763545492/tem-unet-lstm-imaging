import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_heads: int = 4):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.head_dim = hidden_size // attention_heads
        
        assert self.head_dim * attention_heads == hidden_size, \
            "Hidden size must be divisible by attention heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        Q = self.query(x).view(batch_size, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.out(out)
        
        return out, attn_weights


class AttentionLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, 
                 num_layers: int = 2, bidirectional: bool = True,
                 attention_heads: int = 4):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           bidirectional=bidirectional, batch_first=True)
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.attention = SelfAttention(hidden_size * self.num_directions, attention_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        attn_out, attn_weights = self.attention(lstm_out)
        
        h_n = h_n.view(-1, self.num_directions, self.hidden_size)[-1:]
        c_n = c_n.view(-1, self.num_directions, self.hidden_size)[-1:]
        
        return attn_out, (h_n, c_n), attn_weights


class UNetAttentionLSTM(nn.Module):
    def __init__(self, input_channels: int = 1, input_length: int = 30,
                 lstm_hidden_size: int = 128, num_classes: int = 2,
                 attention_heads: int = 4, dropout_p: float = 0.1):
        super().__init__()
        # Dropout1d 无参数，加入后不影响既有 checkpoint 的 state_dict 加载；
        # 推理阶段可通过 enable_dropout 开启以进行 MC Dropout 不确定性量化
        self.drop = nn.Dropout1d(dropout_p)
        
        self.unet = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # 移除第二次池化：30个时间道太短，两次池化(30->15->7)信息损失过大
            # 保留一次池化(30->15)以平衡感受野与分辨率
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        self.attention_lstm = AttentionLSTM(
            input_size=input_channels, 
            hidden_size=lstm_hidden_size,
            attention_heads=attention_heads
        )
        
        self.fusion_conv = nn.Conv1d(lstm_hidden_size * 2 + 256, 256, kernel_size=1)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unet_out = self.unet(x)
        target_len = unet_out.shape[2]
        
        lstm_out, _, attn_weights = self.attention_lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = F.adaptive_avg_pool1d(lstm_out, target_len)
        
        fused = torch.cat([unet_out, lstm_out], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        fused = self.drop(fused)
        
        output = self.decoder(fused)
        
        input_len = x.shape[2]
        output_len = output.shape[2]
        diff = input_len - output_len
        if diff > 0:
            output = F.pad(output, [diff // 2, diff - diff // 2])
        
        return output, attn_weights

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))