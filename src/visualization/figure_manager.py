"""图件管理器

统一管理论文、项目报告、工作总结所需的图件生成与版本控制，
防止原有 fig1-fig8 被覆盖，并自动记录图件用途、说明和元数据。
"""
import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path


class FigureManager:
    """图件版本与用途管理"""

    PAPER_FIGURES = [
        'fig1_network_structure.png',
        'fig2_attention_mechanism.png',
        'fig3_tem_decay_curves.png',
        'fig4_training_loss.png',
        'fig5_domain_adaptation.png',
        'fig6_inversion_comparison.png',
        'fig7_terrain_correction.png',
        'fig8_efficiency_comparison.png',
    ]

    def __init__(self, base_dir: str = 'figures'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.meta_path = self.base_dir / 'figure_metadata.json'
        self.meta = self._load_meta()
        self._protect_paper_figures()

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _protect_paper_figures(self):
        """标记论文核心图件为受保护。"""
        for fig in self.PAPER_FIGURES:
            if fig not in self.meta:
                self.meta[fig] = {
                    'protected': True,
                    'usage': '论文_基于U-Net与LSTM的瞬变电磁数据快速成像方法研究',
                    'created': '2026-08-02',
                    'description': '论文核心图件，禁止覆盖'
                }
        self._save_meta()

    def is_protected(self, filename: str) -> bool:
        entry = self.meta.get(filename, {})
        return entry.get('protected', False)

    def register(self, filename: str, usage: str, description: str = '',
                 overwrite_ok: bool = False) -> Path:
        """注册新图件，自动命名避免冲突。

        Args:
            filename: 目标文件名
            usage: 用途标签，如 '论文' / '项目报告' / '工作总结' / '月报'
            description: 图件说明
            overwrite_ok: 是否允许覆盖非保护图件
        Returns:
            最终保存的 Path
        """
        target = self.base_dir / filename

        if self.is_protected(filename):
            # 自动重命名：fig6_inversion_comparison.png -> fig6_inversion_comparison_report_20261010.png
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            new_name = f"{stem}_{usage}_{datetime.now().strftime('%Y%m%d')}{suffix}"
            target = self.base_dir / new_name
        elif target.exists() and not overwrite_ok:
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            new_name = f"{stem}_v{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}"
            target = self.base_dir / new_name

        return target

    def save_figure(self, source_path: str, filename: str,
                    usage: str, description: str = '',
                    overwrite_ok: bool = False) -> Path:
        """将外部生成的图件保存到图件目录并注册。"""
        target = self.register(filename, usage, description, overwrite_ok)
        shutil.copy2(source_path, target)
        self.meta[target.name] = {
            'protected': False,
            'usage': usage,
            'description': description,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': source_path
        }
        self._save_meta()
        return target

    def list_figures(self, usage: Optional[str] = None) -> List[Dict]:
        """列出图件元数据，可按用途过滤。"""
        results = []
        for fname, meta in self.meta.items():
            fpath = self.base_dir / fname
            if not fpath.exists():
                continue
            if usage is None or meta.get('usage', '') == usage:
                results.append({
                    'filename': fname,
                    'path': str(fpath),
                    'size_kb': fpath.stat().st_size / 1024,
                    **meta
                })
        return results

    def generate_summary(self, output_path: Optional[str] = None) -> str:
        """生成图件清单摘要 Markdown，供报告直接引用。"""
        lines = ['# 图件清单\n']
        categories = {}
        for fname, meta in self.meta.items():
            fpath = self.base_dir / fname
            if not fpath.exists():
                continue
            usage = meta.get('usage', '未分类')
            categories.setdefault(usage, []).append((fname, meta))

        for usage, items in sorted(categories.items()):
            lines.append(f'\n## {usage}\n')
            for fname, meta in items:
                desc = meta.get('description', '')
                lines.append(f"- **{fname}**: {desc}")

        summary = '\n'.join(lines)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
        return summary
