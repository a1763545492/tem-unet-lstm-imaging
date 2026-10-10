"""反演报告自动生成模块

基于 python-docx 自动生成反演成果报告（Word），
同时支持 Markdown 输出；报告自动嵌入图件目录中的图，
图件路径经 FigureManager 校验，不覆盖论文 fig1-fig8。
"""
import os
from datetime import datetime
from typing import Optional, List, Dict

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


class ReportGenerator:
    """项目反演成果报告自动生成器"""

    def __init__(self, project_name: str = '基于U-Net与LSTM的瞬变电磁数据快速成像方法研究',
                 project_no: str = '陕地科创[2025]09号'):
        self.project_name = project_name
        self.project_no = project_no

    # ---------------- Word ----------------
    def generate_word(self, output_path: str,
                      sections: List[Dict],
                      title: Optional[str] = None) -> str:
        """生成 Word 报告。

        Args:
            sections: [{'heading': str, 'level': int,
                        'paragraphs': [str,...],
                        'figures': [{'path': str, 'caption': str, 'width_in': float}],
                        'table': {'headers': [...], 'rows': [[...]]}}]
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        doc = Document()

        # 标题页
        t = doc.add_heading(title or f'{self.project_name}反演成果报告', level=0)
        t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = doc.add_paragraph(f'项目编号：{self.project_no}')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2 = doc.add_paragraph(f'生成日期：{datetime.now().strftime("%Y年%m月%d日")}')
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()

        for sec in sections:
            doc.add_heading(sec.get('heading', ''), level=sec.get('level', 1))
            for para in sec.get('paragraphs', []):
                doc.add_paragraph(para)
            if 'table' in sec and sec['table']:
                tb = sec['table']
                table = doc.add_table(rows=1, cols=len(tb['headers']))
                table.style = 'Light Grid Accent 1'
                for i, h in enumerate(tb['headers']):
                    table.rows[0].cells[i].text = str(h)
                for row in tb['rows']:
                    cells = table.add_row().cells
                    for i, v in enumerate(row):
                        cells[i].text = str(v)
            for fig in sec.get('figures', []):
                if os.path.exists(fig['path']):
                    doc.add_picture(fig['path'],
                                    width=Inches(fig.get('width_in', 5.5)))
                    cap = doc.add_paragraph(fig.get('caption', ''))
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in cap.runs:
                        run.font.size = Pt(9)

        doc.save(output_path)
        return output_path

    # ---------------- Markdown ----------------
    def generate_markdown(self, output_path: str,
                          sections: List[Dict],
                          title: Optional[str] = None) -> str:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        lines = [f'# {title or self.project_name + "反演成果报告"}', '']
        lines.append(f'**项目编号：** {self.project_no}  ')
        lines.append(f'**生成日期：** {datetime.now().strftime("%Y年%m月%d日")}')
        lines.append('\n---\n')
        for sec in sections:
            level = sec.get('level', 1)
            lines.append(f'{"#" * level} {sec.get("heading", "")}\n')
            for para in sec.get('paragraphs', []):
                lines.append(para + '\n')
            if 'table' in sec and sec['table']:
                tb = sec['table']
                lines.append('| ' + ' | '.join(map(str, tb['headers'])) + ' |')
                lines.append('|' + '---|' * len(tb['headers']))
                for row in tb['rows']:
                    lines.append('| ' + ' | '.join(map(str, row)) + ' |')
                lines.append('')
            for fig in sec.get('figures', []):
                if os.path.exists(fig['path']):
                    rel = os.path.relpath(fig['path'],
                                          os.path.dirname(output_path) or '.')
                    lines.append(f'![{fig.get("caption", "")}]({rel})')
                    lines.append(f'*{fig.get("caption", "")}*\n')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return output_path
