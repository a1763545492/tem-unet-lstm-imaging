# -*- coding: utf-8 -*-
"""
将论文Markdown转换为Word文档
符合中文核心期刊排版规范
"""
import os
import subprocess
import sys
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def convert_md_to_docx_with_pandoc(md_file, docx_file, figures_dir):
    """使用pandoc将md转换为docx，并嵌入图片"""
    print(f"使用pandoc转换: {md_file} -> {docx_file}")
    
    # pandoc命令
    cmd = [
        'pandoc',
        md_file,
        '-o', docx_file,
        '--from', 'markdown',
        '--to', 'docx',
        '--standalone',
        '--resource-path', os.path.dirname(os.path.abspath(md_file)),
        '--reference-doc', '',  # 可指定参考模板
        '--toc',  # 生成目录
        '--toc-depth=3',
        '--number-sections',  # 章节自动编号
    ]
    
    # 移除空参数
    cmd = [c for c in cmd if c != '']
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"pandoc警告: {result.stderr}")
    else:
        print("pandoc转换成功")
    
    return docx_file


def format_docx_for_journal(docx_file, figures_dir):
    """按期刊规范格式化Word文档"""
    print("正在按期刊规范格式化文档...")
    
    doc = Document(docx_file)
    
    # 设置页面尺寸（A4）
    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    
    # 设置默认字体（宋体，小四号）
    style = doc.styles['Normal']
    font = style.font
    font.name = 'SimSun'
    font.size = Pt(10.5)  # 五号
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    
    # 段落格式：首行缩进2字符，1.5倍行距
    para_format = style.paragraph_format
    para_format.line_spacing = 1.5
    para_format.first_line_indent = Cm(0.74)  # 约2字符
    
    # 设置标题样式
    heading_styles = {
        'Heading 1': ('黑体', 14, True),   # 三号黑体
        'Heading 2': ('黑体', 12, True),   # 小三号黑体
        'Heading 3': ('黑体', 11, True),   # 四号黑体
    }
    
    for style_name, (font_name, size, bold) in heading_styles.items():
        if style_name in doc.styles:
            heading_style = doc.styles[style_name]
            heading_font = heading_style.font
            heading_font.name = font_name
            heading_font.size = Pt(size)
            heading_font.bold = bold
            heading_font.color.rgb = RGBColor(0, 0, 0)
            heading_style.element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
            heading_style.paragraph_format.first_line_indent = None
            heading_style.paragraph_format.space_before = Pt(12)
            heading_style.paragraph_format.space_after = Pt(6)
    
    # 保存
    doc.save(docx_file)
    print(f"格式化完成: {docx_file}")
    return docx_file


def create_journal_docx_from_scratch(md_file, docx_file, figures_dir):
    """从零创建符合期刊规范的Word文档"""
    print(f"创建期刊规范Word文档: {docx_file}")
    
    doc = Document()
    
    # 页面设置（A4）
    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    
    # 读取markdown内容
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析markdown并写入docx
    lines = content.split('\n')
    i = 0
    in_table = False
    table_rows = []
    
    while i < len(lines):
        line = lines[i]
        
        # 跳过分隔线
        if line.strip() == '---':
            i += 1
            continue
        
        # 标题
        if line.startswith('# ') and not line.startswith('## '):
            title_text = line[2:].strip()
            title_text = clean_markdown(title_text)
            p = doc.add_heading(level=0)
            run = p.add_run(title_text)
            run.font.name = '黑体'
            run.font.size = Pt(16)
            run.font.bold = True
            run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        
        if line.startswith('## '):
            heading_text = clean_markdown(line[3:].strip())
            p = doc.add_heading(level=1)
            run = p.add_run(heading_text)
            run.font.name = '黑体'
            run.font.size = Pt(14)
            run.font.bold = True
            run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            i += 1
            continue
        
        if line.startswith('### '):
            heading_text = clean_markdown(line[4:].strip())
            p = doc.add_heading(level=2)
            run = p.add_run(heading_text)
            run.font.name = '黑体'
            run.font.size = Pt(12)
            run.font.bold = True
            run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            i += 1
            continue
        
        # 表格处理
        if line.startswith('|') and '|' in line[1:]:
            table_rows.append(line)
            i += 1
            while i < len(lines) and lines[i].startswith('|'):
                table_rows.append(lines[i])
                i += 1
            # 解析表格
            if len(table_rows) >= 2:
                add_table_to_doc(doc, table_rows)
            table_rows = []
            continue
        
        # 数学公式块
        if line.strip().startswith('$$'):
            formula_lines = [line.strip().replace('$$', '')]
            i += 1
            while i < len(lines) and '$$' not in lines[i]:
                formula_lines.append(lines[i])
                i += 1
            if i < len(lines):
                formula_lines.append(lines[i].strip().replace('$$', ''))
                i += 1
            formula_text = ' '.join(formula_lines).strip()
            if formula_text:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(formula_text)
                run.font.name = 'Cambria Math'
                run.font.size = Pt(11)
                run.italic = True
            continue
        
        # 图片引用（markdown图片语法）
        if line.strip().startswith('!['):
            # 不处理，图片单独插入
            i += 1
            continue
        
        # 空行
        if not line.strip():
            i += 1
            continue
        
        # 普通段落
        para_text = clean_markdown(line.strip())
        if para_text:
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(0.74)
            p.paragraph_format.line_spacing = 1.5
            
            # 处理加粗
            if '**' in line:
                add_formatted_text(p, line.strip())
            else:
                run = p.add_run(para_text)
                run.font.name = 'SimSun'
                run.font.size = Pt(10.5)
                run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        
        i += 1
    
    # 插入图表（在文档末尾）
    doc.add_page_break()
    p = doc.add_heading('图表', level=1)
    
    figure_files = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
    figure_captions = {
        'fig1_network_structure.png': '图1  U-Net与LSTM融合网络结构图',
        'fig2_attention_mechanism.png': '图2  多头自注意力机制示意图',
        'fig3_tem_decay_curves.png': '图3  TEM衰减曲线与合成数据示例',
        'fig4_training_loss.png': '图4  模型四轮迭代训练Loss收敛曲线',
        'fig5_domain_adaptation.png': '图5  CycleGAN域自适应前后成像效果对比',
        'fig6_inversion_comparison.png': '图6  本文方法与BOSTICK反演结果对比',
        'fig7_terrain_correction.png': '图7  有限元地形校正效果对比',
        'fig8_efficiency_comparison.png': '图8  计算效率与成像精度对比',
    }
    
    for fig_file in figure_files:
        fig_path = os.path.join(figures_dir, fig_file)
        if os.path.exists(fig_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            run.add_picture(fig_path, width=Cm(15))
            
            # 图题
            caption = figure_captions.get(fig_file, fig_file)
            cap_p = doc.add_paragraph()
            cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_run = cap_p.add_run(caption)
            cap_run.font.name = '黑体'
            cap_run.font.size = Pt(10)
            cap_run.bold = True
            cap_run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            
            doc.add_paragraph()  # 空行
    
    doc.save(docx_file)
    print(f"Word文档创建成功: {docx_file}")
    return docx_file


def clean_markdown(text):
    """清理markdown标记"""
    import re
    # 移除加粗
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 移除斜体
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # 移除行内代码
    text = re.sub(r'`(.+?)`', r'\1', text)
    # 移除链接
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text


def add_formatted_text(paragraph, text):
    """添加带格式的文本（处理加粗）"""
    import re
    parts = re.split(r'(\*\*.+?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            run = paragraph.add_run(part)
        run.font.name = 'SimSun'
        run.font.size = Pt(10.5)
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def add_table_to_doc(doc, table_rows):
    """添加表格到文档"""
    # 解析表格行
    rows_data = []
    for row in table_rows:
        if row.strip().startswith('|') and set(row.strip().strip('|')) <= set('-: '):
            continue  # 跳过分隔行
        cells = [c.strip() for c in row.strip('|').split('|')]
        rows_data.append(cells)
    
    if not rows_data:
        return
    
    # 创建表格
    n_cols = len(rows_data[0])
    table = doc.add_table(rows=len(rows_data), cols=n_cols)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    for i, row_data in enumerate(rows_data):
        for j, cell_text in enumerate(row_data):
            if j < n_cols:
                cell = table.cell(i, j)
                cell.text = clean_markdown(cell_text)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.name = 'SimSun'
                        run.font.size = Pt(9)
                        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                        if i == 0:
                            run.bold = True
    
    doc.add_paragraph()  # 表格后空行


def main():
    md_file = '论文_基于U-Net与LSTM的瞬变电磁数据快速成像方法研究.md'
    figures_dir = 'figures'
    
    # 输出文件名
    base_name = os.path.splitext(md_file)[0]
    docx_file = base_name + '.docx'
    
    print("=" * 60)
    print("论文Markdown转Word转换工具")
    print("=" * 60)
    
    if not os.path.exists(md_file):
        print(f"错误: 找不到文件 {md_file}")
        return
    
    # 方法1: 使用python-docx从零创建（推荐，格式更可控）
    print("\n[方法] 使用python-docx创建期刊规范Word文档...")
    create_journal_docx_from_scratch(md_file, docx_file, figures_dir)
    
    # 获取文件大小
    size_kb = os.path.getsize(docx_file) / 1024
    print(f"\n转换完成！")
    print(f"输出文件: {os.path.abspath(docx_file)}")
    print(f"文件大小: {size_kb:.1f} KB")
    
    print("\n" + "=" * 60)
    print("文档规范说明:")
    print("  - 页面: A4 (21×29.7cm)，页边距2.5cm")
    print("  - 正文: 宋体五号(10.5pt)，1.5倍行距，首行缩进2字符")
    print("  - 标题: 黑体(一级三号/二级小三/三级四号)")
    print("  - 图表: 居中，图题黑体小五号加粗")
    print("  - 表格: 三线表样式，宋体小五号")
    print("=" * 60)


if __name__ == '__main__':
    main()
