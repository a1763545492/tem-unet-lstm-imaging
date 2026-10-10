"""Grad-CAM 模型可解释性模块（1D 网络适配版）

对应 2026年8月/9月月报：引入梯度加权类激活映射（Grad-CAM）技术，
将模型决策过程与输入数据特定区域关联，生成反演结果热力图，
直观展示模型关注的地质异常区域，辅助地质解释校准。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class GradCAM1D:
    """面向一维卷积网络的 Grad-CAM 实现。

    对目标卷积层注册前向/反向钩子，计算目标类别得分对特征图的梯度，
    经全局平均池化得到通道权重，加权求和后 ReLU 得到类激活热力图。
    热力图长度与输入衰减曲线时间道一一对应，可定位模型关注的时间道区间。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, input_tensor: torch.Tensor,
                 target_channel: int = 0,
                 target_position: Optional[int] = None,
                 input_length: Optional[int] = None) -> np.ndarray:
        """生成类激活热力图。

        Args:
            input_tensor: 输入张量，形状 (1, C, L)
            target_channel: 目标输出通道（0=电导率, 1=层厚度）
            target_position: 目标空间位置（深度节点索引），None 时取输出全局平均
            input_length: 热力图重采样长度，None 时为输入长度
        Returns:
            cam: 归一化到 [0,1] 的一维热力图，长度 = input_length
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = input_tensor.to(device).requires_grad_(True)

        output = self.model(x)
        if isinstance(output, tuple):
            output = output[0]

        if target_position is None:
            score = output[:, target_channel, :].mean()
        else:
            target_position = min(target_position, output.shape[2] - 1)
            score = output[:, target_channel, target_position].sum()

        self.model.zero_grad()
        score.backward(retain_graph=False)

        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C', 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, L')
        cam = F.relu(cam)

        if input_length is None:
            input_length = input_tensor.shape[-1]
        cam = F.interpolate(cam, size=input_length,
                            mode='linear', align_corners=False).squeeze()

        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def generate_batch(self, inputs: torch.Tensor, target_channel: int = 0) -> np.ndarray:
        """逐样本生成热力图（逐条反传，避免批间梯度串扰）。"""
        cams = []
        for i in range(inputs.shape[0]):
            cams.append(self.generate(inputs[i:i + 1], target_channel))
        return np.array(cams)


def find_target_layer(model: nn.Module, prefer: str = 'auto') -> nn.Module:
    """自动定位适合 Grad-CAM 的目标卷积层。

    prefer:
      - 'decoder': 取融合解码器最后的卷积层（贴近输出，反映成像决策）
      - 'encoder': 取 U-Net 分支最深层卷积（反映中间表征）
      - 'auto': 优先 decoder，找不到则退化为最后一个 Conv1d
    """
    conv_layers: List[nn.Conv1d] = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    if not conv_layers:
        raise ValueError("模型中未找到 Conv1d 层，无法应用 Grad-CAM")

    if prefer in ('auto', 'decoder') and hasattr(model, 'decoder'):
        decoder_convs = [m for m in model.decoder.modules() if isinstance(m, nn.Conv1d)]
        if decoder_convs:
            # 排除 1x1 输出投影层，取最后一个 kernel>1 的卷积
            wide = [m for m in decoder_convs if m.kernel_size[0] > 1]
            return wide[-1] if wide else decoder_convs[-1]
    if prefer == 'encoder' and hasattr(model, 'unet'):
        unet_convs = [m for m in model.unet.modules() if isinstance(m, nn.Conv1d)]
        if unet_convs:
            return unet_convs[-1]
    return conv_layers[-1]


class InterpretabilityAnalyzer:
    """可解释性综合分析：Grad-CAM 热力图 + 注意力权重提取。

    输出供可视化模块绘制论文/报告图件，并供报告生成模块引用。
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        if target_layer is None:
            target_layer = find_target_layer(model, prefer='auto')
        self.gradcam = GradCAM1D(model, target_layer)
        self.model = model

    def analyze(self, curve: Union[np.ndarray, torch.Tensor],
                target_channel: int = 0) -> dict:
        """对单条衰减曲线做完整可解释性分析。

        Returns:
            dict 包含 cam（热力图）、attn_weights（注意力权重矩阵，若有）、
            time_gates_attention（时间道平均注意力）
        """
        if isinstance(curve, np.ndarray):
            curve = torch.from_numpy(curve).float()
        if curve.dim() == 1:
            curve = curve.unsqueeze(0).unsqueeze(0)
        elif curve.dim() == 2:
            curve = curve.unsqueeze(0)

        cam = self.gradcam.generate(curve, target_channel=target_channel,
                                    input_length=curve.shape[-1])

        result = {'cam': cam, 'attn_weights': None, 'time_gates_attention': None}

        self.model.eval()
        with torch.no_grad():
            out = self.model(curve)
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            attn = out[1].detach().cpu().numpy()  # (B, heads, L, L) 或 (B, L, L)
            result['attn_weights'] = attn
            result['time_gates_attention'] = attn.mean(axis=tuple(range(attn.ndim - 1)))
        return result

    def close(self):
        self.gradcam.remove_hooks()
