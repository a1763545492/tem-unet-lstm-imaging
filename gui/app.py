# -*- coding: utf-8 -*-
"""TEM 快速成像可视化软件（Alpha 版）

对应 2026年8月/9月月报：数据加载、参数配置、反演计算、
结果二维显示、三维可视化、数据导出等核心功能。

运行方式：python gui/app.py
依赖：Tkinter（标准库）+ matplotlib + torch（可选，用于加载训练好的模型）
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

plt_rc = {'font.sans-serif': ['SimHei', 'Microsoft YaHei'], 'axes.unicode_minus': False}
matplotlib.rcParams.update(plt_rc)


class TEMApp(tk.Tk):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.title('瞬变电磁数据快速成像系统 v0.9 (Alpha)')
        self.geometry('1200x760')

        self.data = None          # (n_stations, n_gates) 衰减曲线
        self.time_gates = np.logspace(-6, -2, 30)
        self.result_section = None  # 反演断面
        self.model = None

        self._build_menu()
        self._build_layout()

        if '--auto-demo' in sys.argv:
            # 启动后自动加载演示数据并执行反演
            self.after(600, self._auto_demo)

    def _auto_demo(self):
        print('[auto-demo] 生成演示数据...')
        self.make_demo()
        print(f'[auto-demo] 数据形状: {self.data.shape}')
        self.after(300, self._auto_invert)

    def _auto_invert(self):
        print('[auto-demo] 开始反演...')
        self.run_inversion()

    # ---------- UI ----------
    def _build_menu(self):
        menubar = tk.Menu(self)
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label='加载数据(.npy/.csv)', command=self.load_data)
        m_file.add_command(label='加载模型(.pth)', command=self.load_model)
        m_file.add_separator()
        m_file.add_command(label='导出断面图', command=self.export_section_fig)
        m_file.add_command(label='导出结果Excel', command=self.export_excel)
        m_file.add_separator()
        m_file.add_command(label='退出', command=self.destroy)
        menubar.add_cascade(label='文件', menu=m_file)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label='关于', command=lambda: messagebox.showinfo(
            '关于', '基于U-Net与LSTM的瞬变电磁数据快速成像系统\n'
                   '项目编号：陕地科创[2025]09号'))
        menubar.add_cascade(label='帮助', menu=m_help)
        self.config(menu=menubar)

    def _build_layout(self):
        # 左侧参数面板
        left = ttk.LabelFrame(self, text='参数配置', width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(left, text='装置类型').pack(anchor='w', padx=6)
        self.cb_device = ttk.Combobox(left, values=['中心回线', '重叠回线', '大定源回线'],
                                      state='readonly')
        self.cb_device.current(0)
        self.cb_device.pack(fill=tk.X, padx=6, pady=2)

        ttk.Label(left, text='最大深度 (m)').pack(anchor='w', padx=6)
        self.var_maxdepth = tk.StringVar(value='300')
        ttk.Entry(left, textvariable=self.var_maxdepth).pack(fill=tk.X, padx=6, pady=2)

        ttk.Label(left, text='深度节点数').pack(anchor='w', padx=6)
        self.var_nz = tk.StringVar(value='30')
        ttk.Entry(left, textvariable=self.var_nz).pack(fill=tk.X, padx=6, pady=2)

        ttk.Label(left, text='色标').pack(anchor='w', padx=6)
        self.cb_cmap = ttk.Combobox(left, values=['jet', 'viridis', 'RdYlGn_r', 'rainbow'],
                                    state='readonly')
        self.cb_cmap.current(0)
        self.cb_cmap.pack(fill=tk.X, padx=6, pady=2)

        self.var_terrain = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text='应用地形校正', variable=self.var_terrain).pack(anchor='w', padx=6, pady=4)

        ttk.Button(left, text='生成演示数据', command=self.make_demo).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(left, text='开始反演', command=self.run_inversion).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(left, text='Grad-CAM 可解释性', command=self.show_gradcam).pack(fill=tk.X, padx=6, pady=4)

        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.pack(fill=tk.X, padx=6, pady=8)
        self.lbl_status = ttk.Label(left, text='就绪')
        self.lbl_status.pack(anchor='w', padx=6)

        # 右侧 Notebook：二维断面 / 衰减曲线 / 三维
        nb = ttk.Notebook(self)
        nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig2d = Figure(figsize=(7, 5), dpi=100)
        self.ax2d = self.fig2d.add_subplot(111)
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=nb)
        nb.add(self.canvas2d.get_tk_widget(), text='二维断面')

        self.fig_curve = Figure(figsize=(7, 5), dpi=100)
        self.ax_curve = self.fig_curve.add_subplot(111)
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master=nb)
        nb.add(self.canvas_curve.get_tk_widget(), text='衰减曲线')

        self.fig3d = Figure(figsize=(7, 5), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=nb)
        nb.add(self.canvas3d.get_tk_widget(), text='三维可视化')

    # ---------- 数据 ----------
    def set_status(self, text, pct=None):
        self.lbl_status.config(text=text)
        if pct is not None:
            self.progress['value'] = pct
        self.update_idletasks()

    def make_demo(self):
        """生成演示数据：50 个测点，中间含低阻异常体。"""
        from src.forward.tem_forward import TEMForward
        fwd = TEMForward(self.time_gates)
        rng = np.random.default_rng(0)
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
        self.set_status(f'已生成演示数据 {self.data.shape}', 30)
        self.plot_curves()

    def load_data(self):
        path = filedialog.askopenfilename(
            filetypes=[('NumPy', '*.npy'), ('CSV', '*.csv')])
        if not path:
            return
        if path.endswith('.npy'):
            self.data = np.load(path)
        else:
            self.data = np.loadtxt(path, delimiter=',', skiprows=1)
        if self.data.ndim == 3:  # (n, 1, L)
            self.data = self.data[:, 0, :]
        self.set_status(f'已加载 {os.path.basename(path)} {self.data.shape}', 30)
        self.plot_curves()

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[('PyTorch', '*.pth')])
        if not path:
            return
        try:
            import torch
            from src.models.attention_lstm import UNetAttentionLSTM
            model = UNetAttentionLSTM(input_channels=1, input_length=30, num_classes=2)
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict']
                                  if 'model_state_dict' in ckpt else ckpt)
            model.eval()
            self.model = model
            self.set_status(f'模型已加载：{os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('模型加载失败', str(e))

    # ---------- 绘图 ----------
    def plot_curves(self):
        self.ax_curve.clear()
        if self.data is not None:
            for row in self.data[:10]:
                self.ax_curve.semilogy(self.time_gates * 1000, np.abs(row), alpha=0.7)
            self.ax_curve.set_xlabel('延迟时间 t / ms')
            self.ax_curve.set_ylabel('归一化响应')
            self.ax_curve.set_title('衰减曲线（前10个测点）')
            self.ax_curve.grid(True, which='both', alpha=0.3)
        self.canvas_curve.draw()

    def plot_section(self, section):
        self.ax2d.clear()
        maxd = float(self.var_maxdepth.get())
        nz = int(self.var_nz.get())
        nx = section.shape[0]
        x = np.arange(nx) * 20.0
        z = np.linspace(0, maxd, nz)
        im = self.ax2d.pcolormesh(x, z, section.T, cmap=self.cb_cmap.get(),
                                  shading='auto')
        self.ax2d.invert_yaxis()
        self.ax2d.set_xlabel('距离 / m')
        self.ax2d.set_ylabel('深度 / m')
        self.ax2d.set_title('反演电阻率断面')
        self.fig2d.colorbar(im, ax=self.ax2d, label='电阻率 / (Ω·m)')
        self.canvas2d.draw()

    def plot_3d(self, section):
        self.ax3d.clear()
        maxd = float(self.var_maxdepth.get())
        nz = int(self.var_nz.get())
        nx = section.shape[0]
        x = np.arange(nx) * 20.0
        z = np.linspace(0, maxd, nz)
        X, Z = np.meshgrid(x, -z)  # Z向下为负，符合三维显示习惯
        surf = self.ax3d.plot_surface(X, np.zeros_like(X), Z,
                                      facecolors=matplotlib.colormaps[
                                          self.cb_cmap.get()](
                                          (section.T - section.min()) /
                                          (np.ptp(section) + 1e-12)),
                                      rstride=1, cstride=1, shade=False)
        self.ax3d.set_xlabel('X / m')
        self.ax3d.set_ylabel('Y / m')
        self.ax3d.set_zlabel('高程 / m')
        self.ax3d.set_title('三维反演断面（Y方向扩展示意）')
        self.canvas3d.draw()

    # ---------- 反演 ----------
    def run_inversion(self):
        if self.data is None:
            messagebox.showwarning('提示', '请先加载或生成数据')
            return
        threading.Thread(target=self._inversion_worker, daemon=True).start()

    def _inversion_worker(self):
        self.set_status('反演计算中...', 50)
        n = self.data.shape[0]
        nz = int(self.var_nz.get())
        maxd = float(self.var_maxdepth.get())

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
                idx = np.searchsorted(layer_depths, depths)
                idx = np.clip(idx, 0, cond.shape[1] - 1)
                section[i] = 1.0 / np.maximum(cond[i][idx], 1e-3)
        else:
            # 无模型时的 BOSTICK 近似演示路径
            rho = np.abs(self.data)
            rho = rho / rho.max() * 200 + 10
            section = np.zeros((n, nz))
            for i in range(n):
                section[i] = np.interp(np.linspace(0, 1, nz),
                                       np.linspace(0, 1, rho.shape[1]), rho[i])

        self.result_section = section
        self.plot_section(section)
        self.plot_3d(section)
        self.set_status('反演完成', 100)
        print(f'[inversion] 反演完成，断面形状: {section.shape}, '
              f'电阻率范围: {section.min():.1f}~{section.max():.1f} Ω·m')

    # ---------- Grad-CAM ----------
    def show_gradcam(self):
        if self.model is None:
            messagebox.showwarning('提示', '请先加载模型 (.pth)')
            return
        if self.data is None:
            messagebox.showwarning('提示', '请先加载或生成数据')
            return
        from src.interpretability import InterpretabilityAnalyzer
        analyzer = InterpretabilityAnalyzer(self.model)
        res = analyzer.analyze(self.data[0])
        analyzer.close()

        self.ax_curve.clear()
        t_ms = self.time_gates * 1000
        self.ax_curve.semilogy(t_ms, np.abs(self.data[0]), 'b-', label='衰减曲线')
        ax2 = self.ax_curve.twinx()
        ax2.fill_between(t_ms, 0, res['cam'], color='red', alpha=0.35,
                         label='Grad-CAM 权重')
        ax2.set_ylabel('Grad-CAM 权重', color='red')
        self.ax_curve.set_xlabel('延迟时间 t / ms')
        self.ax_curve.set_title('可解释性分析：模型关注区域（红色）')
        self.canvas_curve.draw()
        self.set_status('Grad-CAM 分析完成')

    # ---------- 导出 ----------
    def export_section_fig(self):
        if self.result_section is None:
            messagebox.showwarning('提示', '暂无反演结果')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                            filetypes=[('PNG', '*.png')])
        if path:
            self.fig2d.savefig(path, dpi=600, bbox_inches='tight')
            self.set_status(f'断面图已保存：{path}')

    def export_excel(self):
        if self.result_section is None:
            messagebox.showwarning('提示', '暂无反演结果')
            return
        from src.reporting import DataExporter
        path = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                            filetypes=[('Excel', '*.xlsx')])
        if not path:
            return
        nz = int(self.var_nz.get())
        maxd = float(self.var_maxdepth.get())
        depths = np.linspace(0, maxd, nz)
        exporter = DataExporter()
        exporter.export_section_excel(
            path, np.arange(self.result_section.shape[0]) * 20.0,
            depths, self.result_section.T)
        self.set_status(f'Excel 已导出：{path}')


if __name__ == '__main__':
    app = TEMApp()
    app.mainloop()
