"""面试评估报告相关"""
# 注：主要评估逻辑已集成在 interviewer.py 的 generate_report 方法中
# 此文件用于后续扩展：如本地评分规则、报告持久化等

import os
import re
import json
from datetime import datetime
from config import INTERVIEW_RECORD_DIR


def save_interview_record(conversation_history: list, report: str, metadata: dict = None) -> str:
    """
    保存面试记录到本地文件
    返回保存的文件路径
    """
    os.makedirs(INTERVIEW_RECORD_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_{timestamp}.json"
    filepath = os.path.join(INTERVIEW_RECORD_DIR, filename)

    record = {
        "timestamp": timestamp,
        "conversation": conversation_history,
        "report": report,
        "metadata": metadata or {},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return filepath


def save_report_markdown(report: str) -> str:
    """保存Markdown格式的报告"""
    os.makedirs(INTERVIEW_RECORD_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    filepath = os.path.join(INTERVIEW_RECORD_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath


def save_report_docx(report: str) -> str:
    """
    将Markdown格式的评估报告转换为Word文档(.docx)
    返回保存的文件路径
    """
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    os.makedirs(INTERVIEW_RECORD_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"评估报告_{timestamp}.docx"
    filepath = os.path.join(INTERVIEW_RECORD_DIR, filename)

    doc = Document()

    # 设置默认字体
    style = doc.styles['Normal']
    font = style.font
    font.name = '微软雅黑'
    font.size = Pt(11)

    lines = report.strip().split('\n')
    in_table = False
    table_headers = []
    table_rows = []

    for line in lines:
        line = line.rstrip()

        # 跳过空行
        if not line.strip():
            if in_table and table_rows:
                _add_table(doc, table_headers, table_rows)
                table_headers = []
                table_rows = []
                in_table = False
            doc.add_paragraph()
            continue

        # 一级标题 (# 标题)
        if line.startswith('# ') and not line.startswith('## '):
            if in_table and table_rows:
                _add_table(doc, table_headers, table_rows)
                table_headers = []
                table_rows = []
                in_table = False
            heading = doc.add_heading(line[2:].strip(), level=1)
            heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue

        # 二级标题 (## 标题)
        if line.startswith('## '):
            if in_table and table_rows:
                _add_table(doc, table_headers, table_rows)
                table_headers = []
                table_rows = []
                in_table = False
            doc.add_heading(line[3:].strip(), level=2)
            continue

        # 三级标题 (### 标题)
        if line.startswith('### '):
            if in_table and table_rows:
                _add_table(doc, table_headers, table_rows)
                table_headers = []
                table_rows = []
                in_table = False
            doc.add_heading(line[4:].strip(), level=3)
            continue

        # Markdown表格分隔线 (|---|---|)
        if re.match(r'^\|[\s\-:|]+\|$', line) and in_table:
            continue

        # Markdown表格行 (| 内容 | 内容 |)
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if not in_table:
                table_headers = cells
                in_table = True
            else:
                table_rows.append(cells)
            continue

        # 表格结束后的普通行
        if in_table and table_rows:
            _add_table(doc, table_headers, table_rows)
            table_headers = []
            table_rows = []
            in_table = False

        # 粗体 (**文字**)
        if '**' in line:
            p = doc.add_paragraph()
            _add_rich_text(p, line)
            continue

        # 无序列表 (- 或 * 或 数字.)
        if re.match(r'^[\-\*]\s+', line):
            text = re.sub(r'^[\-\*]\s+', '', line)
            p = doc.add_paragraph(style='List Bullet')
            _add_rich_text(p, text)
            continue

        if re.match(r'^\d+\.\s+', line):
            text = re.sub(r'^\d+\.\s+', '', line)
            p = doc.add_paragraph(style='List Number')
            _add_rich_text(p, text)
            continue

        # 引用 (> )
        if line.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(20)
            run = p.add_run(line[2:].strip())
            run.italic = True
            run.font.color.rgb = RGBColor(100, 100, 100)
            continue

        # 普通段落
        p = doc.add_paragraph()
        _add_rich_text(p, line)

    # 处理末尾可能残留的表格
    if in_table and table_rows:
        _add_table(doc, table_headers, table_rows)

    doc.save(filepath)
    return filepath


def _add_table(doc, headers, rows):
    """添加表格到文档"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'

    # 表头
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(header)
        run.bold = True

    # 表格内容
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            if c_idx < len(headers):
                table.rows[r_idx + 1].cells[c_idx].text = cell_text


def _add_rich_text(paragraph, text):
    """
    处理简单的Markdown富文本：**粗体**、*斜体*、`代码`
    直接追加到给定的paragraph中
    """
    # 用正则分割粗体/普通文本
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            # 再处理内联代码
            subparts = re.split(r'(`[^`]+`)', part)
            for sp in subparts:
                if sp.startswith('`') and sp.endswith('`'):
                    run = paragraph.add_run(sp[1:-1])
                    run.font.name = 'Consolas'
                else:
                    paragraph.add_run(sp)


if __name__ == "__main__":
    # 测试
    test_report = """# 面试评估报告

## 一、总体评价
- 总体评分：**85/100**
- 一句话总结：表现良好，基础扎实

## 二、各维度评分
| 维度 | 评分 | 简评 |
|------|------|------|
| 基础理论 | 88/100 | 掌握较好 |
| RAG能力 | 82/100 | 有实践经验 |

## 三、掌握较好的知识点
- Attention机制
- Transformer结构

## 四、薄弱环节与改进建议
- **LoRA原理**: 需要深入理解
  - 建议: 阅读相关论文
"""
    path = save_report_docx(test_report)
    print(f"Word报告已生成: {path}")
