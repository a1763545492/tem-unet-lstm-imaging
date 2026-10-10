"""二维断面绘图器

提供符合《地球物理学进展》期刊规范的 TEM 反演断面图绘制，
支持单/多剖面对比、钻孔标注、异常体圈定、地形线叠加等。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, SymLogNorm
from typing import Optional, List, Tuple, Dict, Union
import os

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'


class SectionPlotter:
    """瞬变电磁反演断面二维可视化"""

    CMAP_DEFAULT = 'jet'
    CMAP_RESISTIVITY = 'jet'
    CMAP_SECTION = 'RdYlGn_r'

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize

    @staticmethod
    def _format_label(label: str, unit: str) -> str:
        return f"{label} / ({unit})"

    def plot_single_section(self,
                            x: np.ndarray,
                            z: np.ndarray,
                            resistivity: np.ndarray,
                            title: str = '反演电阻率断面',
                            x_label: str = '距离',
                            z_label: str = '深度',
                            resistivity_label: str = '电阻率',
                            x_unit: str = 'm',
                            z_unit: str = 'm',
                            r_unit: str = 'Ω·m',
                            cmap: str = CMAP_DEFAULT,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            terrain: Optional[np.ndarray] = None,
                            boreholes: Optional[List[Tuple[float, str]]] = None,
                            anomalies: Optional[List[Tuple[float, float, float, float, str]]] = None,
                            fig_path: Optional[str] = None) -> plt.Figure:
        """绘制单条电阻率断面。

        Args:
            x: 水平距离数组 (nx,)
            z: 深度数组 (nz,), 向下为正
            resistivity: 电阻率二维数组 (nz, nx)
            terrain: 地形高程 (nx,), 相对地表为0向上正（可选）
            boreholes: [(x_pos, label), ...] 钻孔位置标注
            anomalies: [(x0, z0, w, h, label), ...] 矩形异常体标注
            fig_path: 若给定则保存图片
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if vmin is None:
            vmin = max(resistivity.min(), 0.1)
        if vmax is None:
            vmax = resistivity.max()

        im = ax.pcolormesh(x, z, resistivity, cmap=cmap, shading='auto',
                           vmin=vmin, vmax=vmax)

        # 地形
        if terrain is not None:
            ax.plot(x, terrain, 'k-', linewidth=1.5)
            ax.fill_between(x, terrain, z.min() - 10, color='saddlebrown', alpha=0.5)

        # 钻孔
        if boreholes:
            for bx, bname in boreholes:
                ax.plot([bx, bx], [z.min(), z.max()], 'w--', linewidth=1.5, alpha=0.8)
                ax.plot(bx, z.min() - 5, 'wv', markersize=6)
                ax.text(bx, z.min() - 12, bname, ha='center', fontsize=8,
                        color='white', fontweight='bold')

        # 异常体框
        if anomalies:
            for ax0, az0, w, h, label in anomalies:
                rect = Rectangle((ax0, az0), w, h, fill=False,
                                 edgecolor='yellow', linewidth=2, linestyle='-')
                ax.add_patch(rect)
                ax.text(ax0 + w / 2, az0 + h / 2, label,
                        ha='center', va='center', color='yellow',
                        fontsize=9, fontweight='bold')

        ax.invert_yaxis()
        ax.set_xlabel(self._format_label(x_label, x_unit))
        ax.set_ylabel(self._format_label(z_label, z_unit))
        ax.set_title(title, fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, label=self._format_label(resistivity_label, r_unit))
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()
        if fig_path:
            os.makedirs(os.path.dirname(fig_path) or '.', exist_ok=True)
            plt.savefig(fig_path, dpi=600, bbox_inches='tight')
            plt.close()
        return fig

    def plot_comparison(self,
                        x: np.ndarray,
                        z: np.ndarray,
                        models: List[Tuple[np.ndarray, str]],
                        title: str = '不同方法反演结果对比',
                        fig_path: Optional[str] = None,
                        cmap: str = CMAP_DEFAULT,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cols: Optional[int] = None) -> plt.Figure:
        """多方法并列对比。

        models: [(data_array, label), ...]
        """
        n = len(models)
        cols = cols or min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n > 1 else [axes]

        if vmin is None:
            vmin = max(min(d.min() for d, _ in models), 0.1)
        if vmax is None:
            vmax = max(d.max() for d, _ in models)

        for ax, (data, label) in zip(axes, models):
            im = ax.pcolormesh(x, z, data, cmap=cmap, shading='auto',
                               vmin=vmin, vmax=vmax)
            ax.invert_yaxis()
            ax.set_xlabel('距离 / m')
            ax.set_ylabel('深度 / m')
            ax.set_title(label, fontsize=11)
            plt.colorbar(im, ax=ax, label='电阻率 / (Ω·m)')

        for ax in axes[n:]:
            ax.axis('off')

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.0)
        plt.tight_layout()
        if fig_path:
            os.makedirs(os.path.dirname(fig_path) or '.', exist_ok=True)
            plt.savefig(fig_path, dpi=600, bbox_inches='tight')
            plt.close()
        return fig

    def plot_section_series(self,
                            x_list: List[np.ndarray],
                            z: np.ndarray,
                            resistivity_list: List[np.ndarray],
                            labels: List[str],
                            output_dir: str,
                            prefix: str = 'section',
                            cmap: str = CMAP_DEFAULT,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None) -> List[str]:
        """批量生成多条剖面图件，返回文件路径列表。"""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for x, r, lbl in zip(x_list, resistivity_list, labels):
            fname = f"{prefix}_{lbl}.png"
            fpath = os.path.join(output_dir, fname)
            self.plot_single_section(
                x, z, r, title=f'{lbl} 反演断面',
                cmap=cmap, vmin=vmin, vmax=vmax,
                fig_path=fpath
            )
            paths.append(fpath)
        return paths
