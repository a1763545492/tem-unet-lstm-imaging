# TEM-UnetLSTM 代码库完整说明

**项目**：基于物理约束与不确定性量化的瞬变电磁一维智能反演方法
**仓库**：https://github.com/a1763545492/tem-unet-lstm-imaging
**版本**：v1.0（2026-10）
**作者**：吴传波（陕西地矿集团创新研究院）

---

## 1. 项目结构总览

```
tem-unet-lstm-imaging/
├── src/                          # 核心源码（Python 包）
│   ├── forward/                  # 正演引擎（严格/可微/演示）
│   ├── models/                   # 神经网络架构（本文模型+基线）
│   ├── training/                 # 训练器与物理损失
│   ├── inference/                # MC Dropout 不确定性量化
│   ├── baselines/                # 传统方法（OCCAM/Bostick/对比网络）
│   ├── preprocessing/            # 数据预处理与自动标注
│   ├── visualization/            # 图件生成与保护管理
│   ├── interpretability/         # Grad-CAM 可解释性
│   ├── domain_adaptation/        # CycleGAN 域适应（真实数据桥接）
│   ├── reporting/                # 报告导出
│   └── utils/                    # 并行生成与推理加速
├── scripts/                      # 实验与工具脚本
├── figures/                      # 论文图件（fig1~fig13）
├── output/                       # 实验结果 JSON/数据
├── models/                       # 训练权重（.pth，不入 git）
├── docs/                         # 论文草稿与文档
├── gui/                          # PyQt 可视化界面
└── requirements.txt
```

---

## 2. 核心模块详解

### 2.1 正演引擎 `src/forward/`

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `tem_forward_simpeg.py` | `TEMForwardSimPEG` | **数据生成引擎**：SimPEG 0.25.1 DLF 正演，30 网格阶梯剖面协议（论文唯一数据源） |
| `tem_forward_torch.py` | `TorchDLFTEMForward` | **物理损失引擎**：纯 PyTorch 可微正演，复用 SimPEG DLF 系数，与数据生成一致（误差 <1e-7） |
| `tem_forward.py` | `TEMForward` | **演示/兼容层**：继承 `TEMForwardSimPEG`，保留 GUI 与 auto_labeling 接口 |
| `tem_forward_dlf.py` | DLF 系数工具 | 数字滤波系数加载（key_101_2009 Hankel / key_81_2009 余弦） |
| `terrain_model.py` | `TerrainModel`, `TriangularMesh` | 三维地形建模与响应扰动 |
| `terrain_correction.py` | `FEMTerrainCorrection` | FEM 地形校正 |

**协议红线**：`generate_synthetic_data()` 统一为 **30 网格阶梯剖面**（dz=10m, max_depth=300m, σ∈10^[-2.5,-0.7], 界面 30-120m, SNR 15-30dB, seed 可复现）。任何下游模块不得擅自修改采样范围或标签格式。

### 2.2 网络模型 `src/models/`

| 文件 | 类 | 说明 |
|------|-----|------|
| `unet_lstm.py` | `UNet1D`, `LSTMEncoder`, `UNetLSTM` | 基础 UNet+LSTM 编码器（单池化 30→15） |
| `attention_lstm.py` | `SelfAttention`, `AttentionLSTM`, `UNetAttentionLSTM` | **本文模型**：UNet1D(64/128/256) + BiLSTM(2×128) + 4 头自注意力双分支、1×1 融合 |

**架构要点**：
- 输入：(B, 1, 30) log10 响应（全局归一化）
- 输出：(B, 2, 30) — ch0=log10σ 剖面，ch1=厚度/max_depth 剖面（界面处跳变）
- **单池化**（30→15）：两次池化曾导致界面误差 90m+，严禁回退
- 通道注意力：多头自注意力（非 SE 模块）

### 2.3 训练 `src/training/`

| 文件 | 类 | 说明 |
|------|-----|------|
| `physics_loss.py` | `PhysicsInformedLoss` | **核心创新**：log10 幅值空间 MSE + 自适应权重 w2=l2·l_data/l_physics + 稀疏计算（physics_batch=4, every=2） |
| `amp_trainer.py` | `Trainer`, `AMPEngine`, `TEMDataset` | 混合精度训练引擎，支持早停与 best-val 保存 |

**物理损失统一**：`PhysicsInformedLoss(forward_op=TorchDLFTEMForward)`，与数据生成正演严格一致。旧启发式 `DifferentiableTEMForward`（erf 叠加）已删除。

### 2.4 不确定性量化 `src/inference/`

| 文件 | 函数 | 说明 |
|------|------|------|
| `uncertainty.py` | `enable_dropout`, `predict_with_uncertainty` | MC Dropout（T=50 前向），输出逐节点 std 作为预测分散度 |

**关键发现**：原生 90% 区间覆盖率仅 7.3%（严重欠覆盖），线性方差标定后 44.7% —— 论文已量化此局限并给出改进路线。

### 2.5 传统基线 `src/baselines/`

| 文件 | 类 | 说明 |
|------|-----|------|
| `occam1d.py` | `Occam1D` | 光滑约束反演（Gauss-Newton + 有限差分 Jacobian），max_iter=4 |
| `bostick.py` | `BostickInverter` | Bostick 变换近似反演 |
| `networks.py` | `UNet1D`, `ResNet1D` | 消融对比网络 |

### 2.6 预处理 `src/preprocessing/`

| 文件 | 类 | 说明 |
|------|-----|------|
| `auto_labeling.py` | `AutoLabelingPipeline` | **协议已统一**：委托 `TEMForward.generate_synthetic_data`，标签 (2,30) [log10σ; 厚度/max_depth]，seed 可复现 |
| `data_processor.py` | `DataProcessor` | 实测数据预处理（滤波/归一化） |

### 2.7 可视化 `src/visualization/`

| 文件 | 类 | 说明 |
|------|-----|------|
| `figure_manager.py` | `FigureManager` | **图件保护机制**：论文核心图（fig1~fig8）禁止覆盖，新图自动版本化命名 |
| `section_plotter.py` | `SectionPlotter` | 二维剖面绘制 |
| `volume_plotter.py` | `VolumePlotter` | 三维体绘制（PyVista） |

### 2.8 其他模块

| 模块 | 说明 |
|------|------|
| `interpretability/gradcam.py` | Grad-CAM 1D 时序可解释性分析 |
| `domain_adaptation/cyclegan.py` | CycleGAN 域适应（合成→真实数据桥接，未用于本文结果） |
| `reporting/` | 数据导出与报告生成 |
| `utils/` | 并行数据生成、推理加速 |

---

## 3. 实验脚本 `scripts/`

### 3.1 核心实验（论文结果复现）

| 脚本 | 功能 | 关键参数 | 输出 |
|------|------|---------|------|
| `train_physics.py` | **物理对照实验**（创新点 A） | `--samples 2000 --epochs 60 --physics-batch 4 --physics-every 2 --groups AB` | `models/physics_model.pth`, `output/physics_comparison.json` |
| `ablation_study.py` | **网络消融**（UNet1D/ResNet1D/UNet+Attn+LSTM） | `--samples 2000 --epochs 60` | `output/ablation_results.json` |
| `noise_robustness.py` | **抗噪测试**（5-30dB） | `--repeats 20` | `figures/fig_noise_robustness.png` |
| `borehole_validation.py` | **合成钻孔验证**（100 孔） | `--holes 100 --passes 30` | `output/borehole_validation.json`, `figures/fig_borehole_validation.png` |
| `uncertainty_demo.py` | **不确定性演示** | `--passes 50` | `figures/fig_uncertainty_v*.png` |

### 3.2 验证与工具

| 脚本 | 功能 |
|------|------|
| `validate_torch_forward.py` | 验证 TorchDLFTEMForward 与 SimPEG 一致性（5 项：半空间/三层/随机模型/梯度/计时） |
| `run_pipeline.py` | 全模块冒烟测试（14 项 test，含 GUI 导入检查） |
| `generate_figures.py` / `generate_report_figures.py` | 论文图件批量生成 |
| `redraw_fig8.py` | fig8 实测重绘（效率对比） |
| `md_to_docx.py` / `md_to_docx_v2.py` | Markdown → Word 期刊格式导出 |

---

## 4. 关键实验结果（论文数据）

### 4.1 物理对照实验（2000 样本 × 60 轮）

| 指标 | A 组-纯 MSE | B 组-物理约束 | 变化 |
|------|------------|--------------|------|
| RMSE | 0.4383 | 0.4377 | 持平 |
| 界面深度误差 | 37.26 m | **32.88 m** | **-12%** |
| SSIM | 0.4172 | **0.4802** | **+15%** |

### 4.2 消融实验（2000 样本 × 60 轮）

| 网络 | RMSE | 界面误差 | SSIM |
|------|------|---------|------|
| UNet1D | 0.5426 | 29.01 m | 0.3091 |
| ResNet1D | 0.5061 | 29.55 m | 0.3275 |
| **UNet+Attn+LSTM（本文）** | 0.5177 | 30.30 m | **0.3351** |

### 4.3 抗噪测试（repeats=20）

| SNR | 本文 RMSE | OCCAM-1D RMSE |
|-----|----------|---------------|
| 5 dB | 0.49 | 1.04 |
| 10 dB | 0.46 | 0.98 |
| 15 dB | 0.44 | 0.93 |
| 20 dB | 0.42 | 0.90 |
| 25 dB | 0.41 | 0.88 |
| 30 dB | 0.41 | 0.87 |

### 4.4 合成钻孔验证（100 孔）

| 指标 | 本文 | OCCAM-1D |
|------|------|---------|
| 界面深度误差 | 31.4±21.2 m | 81.8±57.8 m |
| 分层电导率偏差 | ≈0（无系统偏差） | 深部 -0.66 log10σ |
| 界面检出率 | 100% | 100% |

### 4.5 效率（实测，Intel i7-12700H）

| 方法 | 单测点耗时 |
|------|-----------|
| Bostick | 0.24 ms |
| **本文 NN（单次）** | **1.93 ms** |
| 本文 + MC Dropout（T=50） | 100.2 ms |
| OCCAM-1D | 10.0 ms |

---

## 5. 复现指南

### 5.1 环境

```bash
# Python 3.10+, Windows/Linux
pip install torch numpy matplotlib simpeg==0.25.1 libdlf geoana
pip install scikit-learn pyvista PyQt5  # 可选：GUI/可视化
```

### 5.2 快速验证

```bash
# 1. 正演一致性验证（5 项）
python scripts/validate_torch_forward.py

# 2. 全模块冒烟（14 项）
python scripts/run_pipeline.py

# 3. 物理对照实验（约 2 小时）
python scripts/train_physics.py --samples 2000 --epochs 60 --physics-batch 4 --physics-every 2

# 4. 抗噪测试
python scripts/noise_robustness.py --repeats 20

# 5. 钻孔验证
python scripts/borehole_validation.py --holes 100 --passes 30
```

### 5.3 协议红线（严禁修改）

1. **30 网格阶梯剖面**：`generate_synthetic_data` 唯一数据源，任何修改需全链路同步
2. **单池化架构**：UNet 仅一次 MaxPool1d(2)，30→15
3. **log10 全局归一化**：输入 X 与标签 ch0 均用 `log10(σ)`，mu/sd 存 `models/norm_params.json`
4. **物理损失同源**：`PhysicsInformedLoss` 的 `forward_op` 必须是 `TorchDLFTEMForward`
5. **图件保护**：fig1~fig8 禁止覆盖，新图自动版本化

---

## 6. 论文图件映射

| 图号 | 文件名 | 生成脚本 | 内容 |
|------|--------|---------|------|
| fig1 | fig1_architecture.png | 手工绘制 | 网络架构示意图 |
| fig2 | fig2_synthetic_models.png | generate_figures.py | 合成模型与响应示例 |
| fig3 | fig3_training_curves.png | generate_figures.py | 训练收敛曲线 |
| fig4 | fig4_inversion_results.png | generate_figures.py | 反演结果对比（剖面） |
| fig5 | fig5_noise_robustness.png | noise_robustness.py | 抗噪测试 RMSE-SNR |
| fig6 | fig6_uncertainty.png | uncertainty_demo.py | MC Dropout 不确定性 |
| fig7 | fig7_gradcam.png | generate_figures.py | Grad-CAM 可解释性 |
| fig8 | fig8_efficiency_comparison_v*.png | redraw_fig8.py | 效率对比（实测） |
| fig9 | fig9_ablation.png | generate_figures.py | 消融实验对比 |
| fig10 | fig10_borehole_validation.png | borehole_validation.py | 钻孔验证三联图 |
| fig_noise | fig_noise_robustness.png | noise_robustness.py | 抗噪（注册保护版） |
| fig_uncertainty | fig_uncertainty_v20261011223317.png | uncertainty_demo.py | 不确定性（浅部异常） |
| fig_borehole | fig_borehole_validation.png | borehole_validation.py | 钻孔验证统计 |

---

## 7. 版本历史

| 版本 | 日期 | 关键变更 |
|------|------|---------|
| v1.0 | 2026-10-13 | 物理损失统一（严格可微正演）、钻孔验证、抗噪、消融、不确定性、fig8 重绘、学报体例论文 v0.3 |

---

## 8. 引用

```bibtex
@article{wu2026tem,
  title={基于物理约束与不确定性量化的瞬变电磁一维智能反演方法},
  author={吴传波},
  journal={地球物理学报},
  year={2026},
  note={投稿中}
}
```

---

*本文档由代码库自动生成，最后更新：2026-10-13*
