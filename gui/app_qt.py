# -*- coding: utf-8 -*-
"""TEM 快速成像可视化软件 —— PyQt5 版（替代 Tkinter 版）

界面布局：左侧参数面板 + 右侧标签页（二维断面 / 衰减曲线 / 三维可视化）
特性：反演在子线程执行，界面不卡顿；支持 Grad-CAM、图件/Excel 导出。

运行：python gui/app_qt.py
"""
import os
import sys

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel,
                             QComboBox, QLineEdit, QCheckBox, QPushButton,
                             QTabWidget, QProgressBar, QFileDialog,
                             QMessageBox, QAction, QSplitter)
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

STYLE = """
QMainWindow { background: #f5f6fa; }
QGroupBox {
    font-weight: bold; border: 1px solid #dcdde1; border-radius: 6px;
    margin-top: 12px; padding-top: 8px; background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #2f3640; }
QPushButton {
    background: #487eb0; color: white; border: none; border-radius: 5px;
    padding: 8px 12px; font-weight: bold;
}
QPushButton:hover { background: #40739e; }
QPushButton:pressed { background: #33608a; }
QPushButton:disabled { background: #b2bec3; }
QPushButton#secondary { background: #7f8fa6; }
QPushButton#secondary:hover { background: #718093; }
QComboBox, QLineEdit {
    border: 1px solid #dcdde1; border-radius: 4px; padding: 5px; background: white;
}
QTabWidget::pane { border: 1px solid #dcdde1; border-radius: 6px; background: white; }
QTabBar::tab {
    background: #e1e2e6; padding: 8px 18px; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: white; font-weight: bold; }
QProgressBar {
    border: 1px solid #dcdde1; border-radius: 5px; text-align: center; height: 16px;
}
QProgressBar::chunk { background: #44bd32; border-radius: 4px; }
QStatusBar { background: #f5f6fa; }
"""


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, is_3d=False):
        self.fig = Figure(figsize=(7, 5), dpi=100)
        if is_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class InversionWorker(QThread):
    """反演计算子线程"""
    finished_ok = pyqtSignal(np.ndarray)
    failed = pyqtSignal(str)

    def __init__(self, data, model, max_depth, nz, parent=None):
        super().__init__(parent)
        self.data, self.model, self.max_depth, self.nz = data, model, max_depth, nz

    def run(self):
        try:
            n = self.data.shape[0]
            nz, maxd = self.nz, self.max_depth
            if self.model is not None:
                import torch
                with torch.no_grad():
                    x = torch.from_numpy(self.data).float().unsqueeze(1)
                    out = self.model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    cond = out[:, 0, :].numpy()
                depths = np.linspace(0, maxd, nz)
                section = np.zeros((n, nz))
                for i in range(n):
                    layer_depths = np.cumsum(out[i, 1].numpy())
                    layer_depths = layer_depths[layer_depths > 0]
                    idx = np.clip(np.searchsorted(layer_depths, depths),
                                  0, cond.shape[1] - 1)
                    section[i] = 1.0 / np.maximum(cond[i][idx], 1e-3)
            else:
                rho = np.abs(self.data)
                rho = rho / rho.max() * 200 + 10
                section = np.zeros((n, nz))
                for i in range(n):
                    section[i] = np.interp(np.linspace(0, 1, nz),
                                           np.linspace(0, 1, rho.shape[1]), rho[i])
            self.finished_ok.emit(section)
        except Exception as e:
            self.failed.emit(str(e))


class TEMQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('瞬变电磁数据快速成像系统 v1.0')
        self.resize(1280, 780)

        self.data = None
        self.time_gates = np.logspace(-6, -2, 30)
        self.result_section = None
        self.model = None
        self.worker = None

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage('就绪')

    # ---------------- UI ----------------
    def _build_menu(self):
        m_file = self.menuBar().addMenu('文件(&F)')
        for item in [('加载数据 (.npy/.csv)', self.load_data),
                     ('加载模型 (.pth)', self.load_model),
                     None,
                     ('导出断面图', self.export_section_fig),
                     ('导出结果 Excel', self.export_excel),
                     None,
                     ('退出', self.close)]:
            if item is None:
                m_file.addSeparator()
                continue
            text, slot = item
            act = QAction(text, self)
            act.triggered.connect(slot)
            m_file.addAction(act)

        m_help = self.menuBar().addMenu('帮助(&H)')
        act = QAction('关于', self)
        act.triggered.connect(lambda: QMessageBox.about(
            self, '关于', '基于 U-Net 与 LSTM 的瞬变电磁数据快速成像系统 v1.0\n'
                         '项目编号：陕地科创[2025]09号'))
        m_help.addAction(act)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- 左侧参数面板 ----
        left = QWidget()
        left.setMaximumWidth(280)
        left.setMinimumWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        gb = QGroupBox('参数配置')
        grid = QGridLayout(gb)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel('装置类型'), 0, 0)
        self.cb_device = QComboBox()
        self.cb_device.addItems(['中心回线', '重叠回线', '大定源回线'])
        grid.addWidget(self.cb_device, 0, 1)

        grid.addWidget(QLabel('最大深度 (m)'), 1, 0)
        self.ed_maxdepth = QLineEdit('300')
        grid.addWidget(self.ed_maxdepth, 1, 1)

        grid.addWidget(QLabel('深度节点数'), 2, 0)
        self.ed_nz = QLineEdit('30')
        grid.addWidget(self.ed_nz, 2, 1)

        grid.addWidget(QLabel('色标'), 3, 0)
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(['jet', 'viridis', 'RdYlGn_r', 'rainbow'])
        grid.addWidget(self.cb_cmap, 3, 1)

        self.ck_terrain = QCheckBox('应用地形校正')
        grid.addWidget(self.ck_terrain, 4, 0, 1, 2)
        lv.addWidget(gb)

        gb2 = QGroupBox('操作')
        v2 = QVBoxLayout(gb2)
        self.btn_demo = QPushButton('生成演示数据')
        self.btn_demo.setObjectName('secondary')
        self.btn_demo.clicked.connect(self.make_demo)
        self.btn_run = QPushButton('开始反演')
        self.btn_run.clicked.connect(self.run_inversion)
        self.btn_cam = QPushButton('Grad-CAM 可解释性')
        self.btn_cam.setObjectName('secondary')
        self.btn_cam.clicked.connect(self.show_gradcam)
        v2.addWidget(self.btn_demo)
        v2.addWidget(self.btn_run)
        v2.addWidget(self.btn_cam)
        lv.addWidget(gb2)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        lv.addWidget(self.progress)

        self.lbl_info = QLabel('就绪')
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet('color: #636e72; padding: 4px;')
        lv.addWidget(self.lbl_info)
        lv.addStretch()

        splitter.addWidget(left)

        # ---- 右侧标签页 ----
        self.tabs = QTabWidget()
        self.canvas_sec = MplCanvas()
        self.canvas_curve = MplCanvas()
        self.canvas_3d = MplCanvas(is_3d=True)
        self.tabs.addTab(self.canvas_sec, '二维断面')
        self.tabs.addTab(self.canvas_curve, '衰减曲线')
        self.tabs.addTab(self.canvas_3d, '三维可视化')
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _status(self, text, pct=None):
        self.lbl_info.setText(text)
        self.statusBar().showMessage(text)
        if pct is not None:
            self.progress.setValue(pct)

    # ---------------- 数据 ----------------
    def make_demo(self):
        from src.forward.tem_forward import TEMForward
        fwd = TEMForward(self.time_gates)
        curves = []
        for i in range(50):
            x = i * 20.0
            if 400 <= x <= 600:
                conds, ths = [0.01, 0.1, 0.02], [80, 60, 0]
            else:
                conds, ths = [0.01, 0.02, 0.05], [80, 60, 0]
            r = fwd.layered_earth_response(conds, ths)
            r = fwd.add_noise(r, snr=20)
            curves.append(r)
        self.data = np.array(curves)
        self._status(f'已生成演示数据 {self.data.shape}', 30)
        self.plot_curves()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '加载数据', '', 'NumPy (*.npy);;CSV (*.csv)')
        if not path:
            return
        try:
            self.data = (np.load(path) if path.endswith('.npy')
                         else np.loadtxt(path, delimiter=',', skiprows=1))
            if self.data.ndim == 3:
                self.data = self.data[:, 0, :]
            self._status(f'已加载 {os.path.basename(path)} {self.data.shape}', 30)
            self.plot_curves()
        except Exception as e:
            QMessageBox.critical(self, '加载失败', str(e))

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载模型', '',
                                              'PyTorch (*.pth)')
        if not path:
            return
        try:
            import torch
            from src.models.attention_lstm import UNetAttentionLSTM
            model = UNetAttentionLSTM(input_channels=1, input_length=30,
                                      num_classes=2)
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict']
                                  if 'model_state_dict' in ckpt else ckpt)
            model.eval()
            self.model = model
            self._status(f'模型已加载：{os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, '模型加载失败', str(e))

    # ---------------- 绘图 ----------------
    def plot_curves(self):
        ax = self.canvas_curve.ax
        ax.clear()
        if self.data is not None:
            for row in self.data[:10]:
                ax.semilogy(self.time_gates * 1000, np.abs(row), alpha=0.7)
            ax.set_xlabel('延迟时间 t / ms')
            ax.set_ylabel('归一化响应')
            ax.set_title('衰减曲线（前10个测点）')
            ax.grid(True, which='both', alpha=0.3)
        self.canvas_curve.draw()

    def plot_section(self, section):
        ax = self.canvas_sec.ax
        self.canvas_sec.fig.clear()
        ax = self.canvas_sec.fig.add_subplot(111)
        maxd = float(self.ed_maxdepth.text())
        nz = int(self.ed_nz.text())
        x = np.arange(section.shape[0]) * 20.0
        z = np.linspace(0, maxd, nz)
        im = ax.pcolormesh(x, z, section.T, cmap=self.cb_cmap.currentText(),
                           shading='auto')
        ax.invert_yaxis()
        ax.set_xlabel('距离 / m')
        ax.set_ylabel('深度 / m')
        ax.set_title('反演电阻率断面')
        self.canvas_sec.fig.colorbar(im, ax=ax, label='电阻率 / (Ω·m)')
        self.canvas_sec.fig.tight_layout()
        self.canvas_sec.draw()

    def plot_3d(self, section):
        fig = self.canvas_3d.fig
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        maxd = float(self.ed_maxdepth.text())
        nz = int(self.ed_nz.text())
        x = np.arange(section.shape[0]) * 20.0
        z = np.linspace(0, maxd, nz)
        X, Z = np.meshgrid(x, -z)
        norm = (section.T - section.min()) / (np.ptp(section) + 1e-12)
        cmap = matplotlib.colormaps[self.cb_cmap.currentText()]
        ax.plot_surface(X, np.zeros_like(X), Z, facecolors=cmap(norm),
                        rstride=1, cstride=1, shade=False)
        ax.set_xlabel('X / m')
        ax.set_ylabel('Y / m')
        ax.set_zlabel('高程 / m')
        ax.set_title('三维反演断面（Y方向扩展示意）')
        fig.tight_layout()
        self.canvas_3d.draw()

    # ---------------- 反演 ----------------
    def run_inversion(self):
        if self.data is None:
            QMessageBox.warning(self, '提示', '请先加载或生成数据')
            return
        self.btn_run.setEnabled(False)
        self._status('反演计算中...', 50)
        self.worker = InversionWorker(
            self.data, self.model,
            float(self.ed_maxdepth.text()), int(self.ed_nz.text()))
        self.worker.finished_ok.connect(self._on_inversion_done)
        self.worker.failed.connect(self._on_inversion_fail)
        self.worker.start()

    def _on_inversion_done(self, section):
        self.result_section = section
        self.plot_section(section)
        self.plot_3d(section)
        self._status('反演完成', 100)
        self.btn_run.setEnabled(True)

    def _on_inversion_fail(self, msg):
        QMessageBox.critical(self, '反演失败', msg)
        self._status('反演失败', 0)
        self.btn_run.setEnabled(True)

    # ---------------- Grad-CAM ----------------
    def show_gradcam(self):
        if self.model is None or self.data is None:
            QMessageBox.warning(self, '提示', '请先加载模型 (.pth) 并生成/加载数据')
            return
        try:
            from src.interpretability import InterpretabilityAnalyzer
            analyzer = InterpretabilityAnalyzer(self.model)
            res = analyzer.analyze(self.data[0])
            analyzer.close()

            fig = self.canvas_curve.fig
            fig.clear()
            ax = fig.add_subplot(111)
            t_ms = self.time_gates * 1000
            ax.semilogy(t_ms, np.abs(self.data[0]), 'b-', label='衰减曲线')
            ax2 = ax.twinx()
            ax2.fill_between(t_ms, 0, res['cam'], color='red', alpha=0.35,
                             label='Grad-CAM 权重')
            ax2.set_ylabel('Grad-CAM 权重', color='red')
            ax.set_xlabel('延迟时间 t / ms')
            ax.set_title('可解释性分析：模型关注区域（红色）')
            fig.tight_layout()
            self.canvas_curve.draw()
            self.tabs.setCurrentWidget(self.canvas_curve)
            self._status('Grad-CAM 分析完成')
        except Exception as e:
            QMessageBox.critical(self, 'Grad-CAM 失败', str(e))

    # ---------------- 导出 ----------------
    def export_section_fig(self):
        if self.result_section is None:
            QMessageBox.warning(self, '提示', '暂无反演结果')
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出断面图', '',
                                              'PNG (*.png)')
        if path:
            self.canvas_sec.fig.savefig(path, dpi=600, bbox_inches='tight')
            self._status(f'断面图已保存：{path}')

    def export_excel(self):
        if self.result_section is None:
            QMessageBox.warning(self, '提示', '暂无反演结果')
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出 Excel', '',
                                              'Excel (*.xlsx)')
        if not path:
            return
        try:
            from src.reporting import DataExporter
            depths = np.linspace(0, float(self.ed_maxdepth.text()),
                                 int(self.ed_nz.text()))
            DataExporter().export_section_excel(
                path, np.arange(self.result_section.shape[0]) * 20.0,
                depths, self.result_section.T)
            self._status(f'Excel 已导出：{path}')
        except Exception as e:
            QMessageBox.critical(self, '导出失败', str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setFont(QFont('Microsoft YaHei', 9))
    win = TEMQtApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
