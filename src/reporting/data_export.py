"""反演结果数据导出模块

按项目约定格式导出 Excel：X, Y, Z_surface, Depth, Z_target, Resistivity
深度约定：Depth = Z_surface - Z_target（向下为正）
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import os


class DataExporter:
    """反演结果标准化导出器"""

    COLUMNS = ['X', 'Y', 'Z_surface', 'Depth', 'Z_target', 'Resistivity']

    @staticmethod
    def build_dataframe(x: np.ndarray, y: np.ndarray,
                        z_surface: np.ndarray,
                        z_target: np.ndarray,
                        resistivity: np.ndarray) -> pd.DataFrame:
        """构建标准化 DataFrame。

        Args:
            x, y: 平面坐标
            z_surface: 地表高程
            z_target: 目标标高（海拔）
            resistivity: 反演电阻率值
        """
        depth = z_surface - z_target  # 向下为正的深度
        df = pd.DataFrame({
            'X': x,
            'Y': y,
            'Z_surface': z_surface,
            'Depth': depth,
            'Z_target': z_target,
            'Resistivity': resistivity
        })
        return df

    def export_excel(self, df: pd.DataFrame, output_path: str,
                     sheet_name: str = '反演结果',
                     extra_sheets: Optional[Dict[str, pd.DataFrame]] = None) -> str:
        """导出 Excel（.xlsx），支持附加工作表（如参数表、统计表）。"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            if extra_sheets:
                for name, sdf in extra_sheets.items():
                    sdf.to_excel(writer, sheet_name=name[:31], index=False)
        return output_path

    def export_section_excel(self, output_path: str,
                             distances: np.ndarray, depths: np.ndarray,
                             resistivity_section: np.ndarray,
                             line_id: Union[str, int] = 1,
                             x_coords: Optional[np.ndarray] = None,
                             y_coords: Optional[np.ndarray] = None,
                             elevations: Optional[np.ndarray] = None) -> str:
        """导出单条剖面为长表 Excel。

        Args:
            distances: 剖面距数组 (nx,)
            depths: 深度数组 (nz,)，向下为正
            resistivity_section: 电阻率断面 (nz, nx)
            x_coords/y_coords/elevations: 测点真实坐标与地表高程（可选，缺省用剖面距/0填充）
        """
        nx = len(distances)
        nz = len(depths)
        if x_coords is None:
            x_coords = distances
        if y_coords is None:
            y_coords = np.full(nx, float(line_id))
        if elevations is None:
            elevations = np.zeros(nx)

        rows = []
        for i in range(nx):
            for j in range(nz):
                z_target = elevations[i] - depths[j]
                rows.append({
                    'X': x_coords[i],
                    'Y': y_coords[i],
                    'Z_surface': elevations[i],
                    'Depth': depths[j],
                    'Z_target': z_target,
                    'Resistivity': resistivity_section[j, i]
                })
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        return self.export_excel(df, output_path, sheet_name=f'剖面{line_id}')

    def export_decay_curves_csv(self, output_path: str,
                                time_gates: np.ndarray,
                                curves: np.ndarray,
                                station_ids: Optional[List] = None) -> str:
        """导出衰减曲线集为 CSV，供论文/报告附表使用。"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        n = curves.shape[0]
        if station_ids is None:
            station_ids = list(range(1, n + 1))
        data = {'Station': station_ids}
        for k, t in enumerate(time_gates):
            data[f't={t * 1000:.3f}ms'] = curves[:, k]
        pd.DataFrame(data).to_csv(output_path, index=False, encoding='utf-8-sig')
        return output_path
