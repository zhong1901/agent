"""简历解析模块 - 提取关键信息用于个性化提问"""
import os
import re
from typing import Dict, Optional
from io import BytesIO


def parse_resume(file_path: str) -> Dict:
    """
    解析简历文件，提取关键信息
    支持PDF和DOCX格式
    返回: {
        "skills": "...",
        "projects": "...",
        "work_experience": "...",
        "education": "...",
        "full_text": "..."
    }
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        text = _extract_pdf_text(file_path)
    elif ext in ['.docx', '.doc']:
        text = _extract_docx_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    return _extract_info(text)


def parse_resume_from_bytes(file_bytes: bytes, filename: str) -> Dict:
    """从文件字节流解析简历（用于Gradio上传）"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.pdf':
        import pdfplumber
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif ext in ['.docx', '.doc']:
        from docx import Document
        doc = Document(BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    return _extract_info(text)


def _extract_pdf_text(file_path: str) -> str:
    import pdfplumber
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n"
    return text


def _extract_docx_text(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def _extract_info(text: str) -> Dict:
    """
    从简历文本中提取关键信息
    使用关键词匹配 + 正则的简单提取策略
    """
    text = text.strip()

    # 提取技能栈
    skills = _extract_section(text, ["技能", "专业技能", "技术栈", "掌握技能", "技能特长", "IT技能"])

    # 提取项目经历
    projects = _extract_section(text, ["项目经历", "项目经验", "主要项目", "项目介绍"])

    # 提取工作经历
    work_experience = _extract_section(text, ["工作经历", "工作经验", "职业经历", "实习经历"])

    # 提取教育背景
    education = _extract_section(text, ["教育背景", "教育经历", "学历", "教育"])

    # 如果提取不到，给个简短的全文摘要（前500字）
    if not any([skills, projects, work_experience, education]):
        summary = text[:800] + ("..." if len(text) > 800 else "")
        skills = summary
        projects = summary
        work_experience = summary
        education = summary

    return {
        "skills": skills[:500] if skills else "未提取到明确的技能信息",
        "projects": projects[:1000] if projects else "未提取到明确的项目经历",
        "work_experience": work_experience[:800] if work_experience else "未提取到明确的工作经历",
        "education": education[:300] if education else "未提取到明确的教育背景",
        "full_text": text[:2000],  # 截断，避免过长
    }


def _extract_section(text: str, keywords: list) -> str:
    """
    根据关键词提取简历中的某个section
    策略：找到关键词所在行，然后取到下一个大section标题之前的内容
    """
    lines = text.split('\n')

    # 找section起始位置
    start_idx = None
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line and len(line.strip()) < 20:  # 标题行不会太长
                start_idx = i + 1
                break
        if start_idx is not None:
            break

    if start_idx is None:
        return ""

    # 找section结束位置（下一个标题）
    end_idx = len(lines)
    section_titles = [
        "教育", "工作", "项目", "技能", "实习", "自我评价", "获奖", "证书",
        "语言", "兴趣", "联系方式", "基本信息", "个人信息", "求职意向",
        "校园经历", "在校经历", "科研", "论文", "专利", "培训"
    ]

    for i in range(start_idx, min(start_idx + 30, len(lines))):
        line = lines[i].strip()
        # 判断是否是新的section标题
        if len(line) < 15 and any(
            t in line for t in section_titles
        ) and i > start_idx:
            # 确保不是当前section的子标题
            if not any(kw in line for kw in keywords):
                end_idx = i
                break

    section_text = "\n".join(lines[start_idx:end_idx]).strip()
    # 清理空行
    section_text = re.sub(r'\n{3,}', '\n\n', section_text)
    return section_text
