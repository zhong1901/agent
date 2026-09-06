import os
import re
import pdfplumber
from typing import List, Dict, Tuple


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    用pdfplumber提取PDF文本，返回完整文本和按页信息列表
    pages_info: [{"page_num": 1, "text": "...", "char_count": 100}, ...]
    """
    full_text = ""
    pages_info = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = clean_text(text)
            full_text += text + "\n\n"
            pages_info.append({
                "page_num": i,
                "text": text,
                "char_count": len(text),
            })

    return full_text.strip(), pages_info


def clean_text(text: str) -> str:
    """清洗PDF提取的文本"""
    # 去除多余空白
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除多余空行（保留段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除页码行（常见模式）
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # 去除页眉页脚常见模式
    text = re.sub(r'^\s*第\s*\d+\s*页.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page\s*\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # 首尾去空白
    text = text.strip()
    return text


def categorize_pdf(filename: str) -> List[str]:
    """
    根据PDF文件名自动分类，返回标签列表
    分类标签：基础理论, RAG, 微调, 分布式训练, 推理优化, 工程实践, 评测
    """
    name = filename.lower()
    categories = []

    # RAG相关
    rag_keywords = ['rag', '检索增强', '向量库', '文档对话', '知识库', '版面分析', '分块', '召回', 'graph rag']
    if any(kw in name for kw in rag_keywords):
        categories.append('RAG')

    # 微调相关
    finetune_keywords = ['微调', 'lora', 'peft', 'adapter', 'prompting', 'sft', '强化学习', 'rlhf']
    if any(kw in name for kw in finetune_keywords):
        categories.append('微调')

    # 分布式训练相关
    dist_keywords = ['分布式', '并行', 'deepspeed', 'accelerate', 'zero', 'dataparallel', 'pipeline']
    if any(kw in name for kw in dist_keywords):
        categories.append('分布式训练')

    # 推理优化
    inference_keywords = ['推理', '显存', '量化', 'kv cache', '推理面']
    if any(kw in name for kw in inference_keywords):
        categories.append('推理优化')

    # 基础理论
    base_keywords = ['基础面', '进阶面', 'attention', 'transformer', '激活函数', 'layer norm',
                     '损失函数', '相似度', '归一化', '训练经验', '训练集']
    if any(kw in name for kw in base_keywords):
        categories.append('基础理论')

    # 工程实践
    eng_keywords = ['langchain', 'lang chain', '工程', '实战', '关键痛点']
    if any(kw in name for kw in eng_keywords):
        categories.append('工程实践')

    # 评测相关
    eval_keywords = ['评测', '评估']
    if any(kw in name for kw in eval_keywords):
        categories.append('评测')

    # 预训练
    pretrain_keywords = ['预训练', 'pretrain', '增量']
    if any(kw in name for kw in pretrain_keywords):
        categories.append('预训练')

    # 默认兜底
    if not categories:
        categories.append('其他')

    return categories


def list_pdf_files(pdf_dir: str) -> List[str]:
    """列出目录下所有PDF文件路径"""
    pdf_files = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(pdf_dir, filename))
    return sorted(pdf_files)
