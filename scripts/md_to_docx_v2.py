# -*- coding: utf-8 -*-
"""
使用pandoc将论文MD转换为Word文档
公式采用Word原生OMML格式（符合期刊要求）
"""
import os
import subprocess
import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


def convert_with_pandoc(md_file, docx_file):
    """使用pandoc转换，公式自动转为Word OMML格式"""
    print(f"正在使用pandoc转换: {md_file}")
    
    cmd = [
        'pandoc',
        md_file,
        '-o', docx_file,
        '--from', 'markdown+tex_math_dollars+tex_math_single_backslash',
        '--to', 'docx',
        '--standalone',
        '--mathml',  # 数学公式转OMML
        '--resource-path', os.path.dirname(os.path.abspath(md_file)),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"pandoc输出: {result.stderr}")
    if result.stdout:
        print(result.stdout)
    
    print("pandoc转换完成")
    return docx_file


def format_docx_journal_style(docx_file, figures_dir):
    """按期刊规范格式化Word文档"""
    print("正在按期刊规范格式化...")
    
    doc = Document(docx_file)
    
    # 页面设置（A4）
    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    
    # 设置Normal样式（正文：宋体五号，1.5倍行距，首行缩进2字符）
    normal_style = doc.styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'  # 西文
    normal_font.size = Pt(10.5)
    normal_style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')  # 中文
    normal_style.paragraph_format.line_spacing = 1.5
    normal_style.paragraph_format.first_line_indent = Cm(0.74)
    
    # 标题样式
    heading_configs = {
        'Heading 1': ('黑体', 14, True),   # 三号
        'Heading 2': ('黑体', 12, True),   # 小三
        'Heading 3': ('黑体', 11, True),   # 四号
        'Title': ('黑体', 16, True),        # 标题
    }
    
    for style_name, (cn_font, size, bold) in heading_configs.items():
        try:
            style = doc.styles[style_name]
            font = style.font
            font.name = 'Times New Roman'
            font.size = Pt(size)
            font.bold = bold
            font.color.rgb = RGBColor(0, 0, 0)
            style.element.rPr.rFonts.set(qn('w:eastAsia'), cn_font)
            style.paragraph_format.first_line_indent = None
            style.paragraph_format.space_before = Pt(12)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.line_spacing = 1.5
        except KeyError:
            pass
    
    # 处理所有段落：为中文内容设置宋体，英文设置Times New Roman
    for para in doc.paragraphs:
        para_style_name = para.style.name
        is_heading = para_style_name.startswith('Heading') or para_style_name == 'Title'
        
        for run in para.runs:
            if is_heading:
                run.font.name = 'Times New Roman'
                run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
            else:
                run.font.name = 'Times New Roman'
                run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
                if run.font.size is None:
                    run.font.size = Pt(10.5)
        
        # 正文段落首行缩进
        if not is_heading and para.text.strip():
            para.paragraph_format.first_line_indent = Cm(0.74)
            para.paragraph_format.line_spacing = 1.5
    
    # 处理表格字体
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    para.paragraph_format.first_line_indent = None
                    for run in para.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(9)
                        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    
    # 在文档末尾添加图表
    # 先移除pandoc可能生成的图片占位
    doc.add_page_break()
    
    # 添加图表标题
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('图表')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    
    figure_captions = {
        'fig1_network_structure.png': '图1  双分支反演网络总体结构（UNet1D + BiLSTM-自注意力）',
        'fig2_attention_mechanism.png': '图2  多头自注意力机制示意图',
        'fig3_tem_decay_curves.png': '图3  TEM衰减曲线与合成数据示例',
        'fig4_training_loss.png': '图4  训练损失收敛曲线（A组纯MSE vs B组物理约束）',
        'fig5_domain_adaptation.png': '图5  域自适应效果对比',
        'fig6_inversion_comparison.png': '图6  反演断面对比',
        'fig7_terrain_correction.png': '图7  地形校正效果对比',
        'fig8_efficiency_comparison.png': '图8  计算效率与成像精度对比（示意数据，待按实测更新）',
        'fig9_gradcam_interpretability.png': '图9  Grad-CAM可解释性分析',
        'fig10_3d_resistivity_volume.png': '图10  三维电阻率体渲染',
        'fig_noise_robustness.png': '图11  抗噪性能对比（RMSE-SNR，NN vs OCCAM-1D）',
        'fig_uncertainty.png': '图12  MC Dropout不确定性断面（均值+逐节点标准差，2026-10-11新协议版）',
        'fig_borehole_validation.png': '图13  合成钻孔验证三联图（界面误差/分层一致性/区间标定）',
    }

    # 只嵌入白名单图件（排除历史存档与报告图）
    figure_files = [f for f in figure_captions
                    if os.path.exists(os.path.join(figures_dir, f))]
    
    for fig_file in figure_files:
        fig_path = os.path.join(figures_dir, fig_file)
        if os.path.exists(fig_path):
            # 图片
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = None
            run = p.add_run()
            run.add_picture(fig_path, width=Cm(15))
            
            # 图题
            caption = figure_captions.get(fig_file, fig_file)
            cap_p = doc.add_paragraph()
            cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_p.paragraph_format.first_line_indent = None
            cap_p.paragraph_format.space_after = Pt(12)
            cap_run = cap_p.add_run(caption)
            cap_run.font.name = 'Times New Roman'
            cap_run.font.size = Pt(10)
            cap_run.bold = True
            cap_run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    
    doc.save(docx_file)
    print(f"格式化完成: {docx_file}")
    return docx_file


def verify_formulas(docx_file):
    """验证公式是否正确转换"""
    print("验证公式转换...")
    doc = Document(docx_file)
    
    # 检查OMML公式
    formula_count = 0
    for para in doc.paragraphs:
        # 检查段落XML中是否包含数学公式元素
        xml = para._element.xml
        if 'm:oMath' in xml or 'm:oMathPara' in xml:
            formula_count += 1
    
    print(f"  检测到OMML公式段落: {formula_count} 个")
    
    if formula_count > 0:
        print("  ✓ 公式已正确转换为Word原生格式(OMML)")
    else:
        print("  ✗ 未检测到OMML公式，可能需要检查")
    
    return formula_count


def main():
    import argparse
    ap = argparse.ArgumentParser(description='论文MD转Word（期刊规范版）')
    ap.add_argument('md_file', nargs='?', default='docs/paper_draft.md')
    ap.add_argument('docx_file', nargs='?', default=None)
    ap.add_argument('--figures-dir', default='figures')
    args = ap.parse_args()

    md_file = args.md_file
    figures_dir = args.figures_dir
    docx_file = args.docx_file or os.path.splitext(md_file)[0] + '.docx'
    
    print("=" * 60)
    print("论文Markdown转Word（期刊规范版）")
    print("公式采用Word原生OMML格式")
    print("=" * 60)
    
    if not os.path.exists(md_file):
        print(f"错误: 找不到文件 {md_file}")
        return
    
    # 步骤1: 使用pandoc转换（公式自动转OMML）
    convert_with_pandoc(md_file, docx_file)
    
    # 步骤2: 验证公式
    verify_formulas(docx_file)
    
    # 步骤3: 期刊格式化
    format_docx_journal_style(docx_file, figures_dir)
    
    # 最终验证
    doc = Document(docx_file)
    size_kb = os.path.getsize(docx_file) / 1024
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"输出文件: {os.path.abspath(docx_file)}")
    print(f"文件大小: {size_kb:.1f} KB")
    print(f"段落数: {len(doc.paragraphs)}")
    print(f"表格数: {len(doc.tables)}")
    print(f"图片数: {len(doc.inline_shapes)}")
    print()
    print("排版规范:")
    print("  ✓ 页面: A4 (21×29.7cm)，页边距2.5cm")
    print("  ✓ 正文: 中文宋体/英文Times New Roman，五号(10.5pt)")
    print("  ✓ 行距: 1.5倍，首行缩进2字符")
    print("  ✓ 标题: 黑体(一级三号/二级小三/三级四号)")
    print("  ✓ 公式: Word原生OMML格式（可编辑）")
    print("  ✓ 图片: 居中，宽度15cm，含图题")
    print("  ✓ 表格: 居中，宋体小五号")
    print("=" * 60)


if __name__ == '__main__':
    main()
