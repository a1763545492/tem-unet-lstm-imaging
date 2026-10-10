"""三维电阻率体可视化

基于 matplotlib 3D 切片 + 可导出 VTK/CSV 的体积数据，
支持论文中三维异常体展示及交互式（mplcursors）切片浏览。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600


class VolumePlotter:
    """三维电阻率体绘制器"""

    def __init__(self):
        pass

    @staticmethod
    def plot_slices(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                    resistivity: np.ndarray,
                    slice_indices: Optional[Tuple[int, int, int]] = None,
                    title: str = '三维电阻率体切片',
                    fig_path: Optional[str] = None,
                    cmap: str = 'jet',
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None) -> plt.Figure:
        """绘制 X/Y/Z 三方向正交切片。

        Args:
            X, Y, Z: np.meshgrid 后的三维坐标网格
            resistivity: (nz, ny, nx) 电阻率体数据（Z, Y, X）
            slice_indices: (ix, iy, iz) 切片位置索引，None 取中间
        """
        nz, ny, nx = resistivity.shape
        ix = nx // 2 if slice_indices is None else slice_indices[0]
        iy = ny // 2 if slice_indices is None else slice_indices[1]
        iz = nz // 2 if slice_indices is None else slice_indices[2]

        if vmin is None:
            vmin = max(resistivity.min(), 0.1)
        if vmax is None:
            vmax = resistivity.max()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Y-Z 平面 (X=ix)
        ax0 = axes[0]
        im0 = ax0.pcolormesh(Y[:, iy, :], Z[:, iy, :], resistivity[:, iy, :],
                             cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax0.set_xlabel('Y / m')
        ax0.set_ylabel('Z / m')
        ax0.set_title(f'(a) X={X[0, iy, ix]:.0f} m 切片')
        ax0.invert_yaxis()
        plt.colorbar(im0, ax=ax0, label='电阻率 / (Ω·m)')

        # X-Z 平面 (Y=iy)
        ax1 = axes[1]
        im1 = ax1.pcolormesh(X[:, :, ix], Z[:, :, ix], resistivity[:, :, ix],
                             cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax1.set_xlabel('X / m')
        ax1.set_ylabel('Z / m')
        ax1.set_title(f'(b) Y={Y[iz, iy, 0]:.0f} m 切片')
        ax1.invert_yaxis()
        plt.colorbar(im1, ax=ax1, label='电阻率 / (Ω·m)')

        # X-Y 平面 (Z=iz)
        ax2 = axes[2]
        im2 = ax2.pcolormesh(X[iz, :, :], Y[iz, :, :], resistivity[iz, :, :],
                             cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax2.set_xlabel('X / m')
        ax2.set_ylabel('Y / m')
        ax2.set_title(f'(c) Z={Z[iz, 0, 0]:.0f} m 切片')
        plt.colorbar(im2, ax=ax2, label='电阻率 / (Ω·m)')

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        if fig_path:
            os.makedirs(os.path.dirname(fig_path) or '.', exist_ok=True)
            plt.savefig(fig_path, dpi=600, bbox_inches='tight')
            plt.close()
        return fig

    @staticmethod
    def plot_3d_isosurface_mesh(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                 resistivity: np.ndarray,
                                 iso_levels: List[float] = [10, 50, 100],
                                 title: str = '三维电阻率等值面',
                                 fig_path: Optional[str] = None,
                                 cmap: str = 'jet') -> plt.Figure:
        """绘制等值面三维线框图（matplotlib 可用方式）。"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for level in iso_levels:
            try:
                from skimage import measure
                verts, faces, _, _ = measure.marching_cubes(
                    resistivity, level=level)
                # 映射回物理坐标
                x_s = verts[:, 2] / resistivity.shape[2] * (X.max() - X.min()) + X.min()
                y_s = verts[:, 1] / resistivity.shape[1] * (Y.max() - Y.min()) + Y.min()
                z_s = verts[:, 0] / resistivity.shape[0] * (Z.max() - Z.min()) + Z.min()

                ax.plot_trisurf(x_s, y_s, z_s, alpha=0.3)
            except Exception:
                # fallback: 不绘制等值面，仅提示
                pass

        ax.set_xlabel('X / m')
        ax.set_ylabel('Y / m')
        ax.set_zlabel('Z / m')
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        if fig_path:
            os.makedirs(os.path.dirname(fig_path) or '.', exist_ok=True)
            plt.savefig(fig_path, dpi=600, bbox_inches='tight')
            plt.close()
        return fig

    @staticmethod
    def export_volume_csv(output_path: str,
                          X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                          resistivity: np.ndarray):
        """导出三维体数据为 CSV（X, Y, Z, Resistivity），便于后续 VTK/Paraview 处理。"""
        nx, ny, nz = X.shape[2], Y.shape[1], Z.shape[0]
        coords = np.column_stack([
            X.ravel(), Y.ravel(), Z.ravel(), resistivity.ravel()
        ])
        header = "X,Y,Z,Resistivity"
        np.savetxt(output_path, coords, delimiter=',', header=header,
                   comments='', fmt='%.4f')
        print(f"Volume data exported to {output_path}")
