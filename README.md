# 🤖 AI大模型面试模拟Agent

> 基于 RAG 的大模型应用开发技术面试模拟系统，AI 扮演资深面试官，题目来自 53 份精选面试题资料。

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/your-username/your-space-name)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://gradio.app/)

## ✨ 功能特性

- 🎯 **分级面试** - 大厂(3轮)/中厂(2轮)/小厂(1轮) 不同面试流程
- 📚 **RAG 知识增强** - 基于 53 份 PDF 面试题资料，问题有依据
- 📝 **简历个性化** - 上传简历，面试官针对你的项目经历深挖
- 📊 **详细评估报告** - 每题含「问题+回答+参考答案+得分」，含录取结论
- 📄 **Word 报告下载** - 一键导出 .docx 格式评估报告
- 👥 **多会话隔离** - 支持多人同时访问，会话数据互不干扰
- 🌐 **一键部署** - 支持 HuggingFace Spaces 免费部署

## 🏗️ 技术栈

| 层级 | 技术选型 |
|------|----------|
| LLM | DeepSeek-V4-Flash (OpenAI 兼容 API) |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 向量库 | ChromaDB |
| 框架 | LangChain |
| 前端 | Gradio |
| 部署 | HuggingFace Spaces |

## 🚀 快速开始

### 本地运行

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd agent-hr

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 4. 运行
python app.py
```

### 部署到 HuggingFace Spaces

#### 方式一：通过网页创建（最简单）

1. 前往 [HuggingFace Spaces](https://huggingface.co/spaces)
2. 点击 **Create new Space**
3. 填写信息：
   - **Space name**: 你的空间名称
   - **Space SDK**: 选 **Gradio**
   - **Space hardware**: 选 **CPU basic · 2 vCPU · 16GB RAM**（免费）
4. 点击 **Create Space**
5. 把项目代码上传到 Space（可以用网页上传或 git push）
6. 在 Space 的 **Settings** → **Variables and secrets** 中添加：
   ```
   Name: DEEPSEEK_API_KEY
   Value: sk-你的key
   ```
7. 等待构建完成，访问 Space 链接即可使用

#### 方式二：通过 git 部署

```bash
# 1. 安装 git-lfs（用于大文件，但PDF不大可以不用）
# git lfs install

# 2. 添加 Space 远程仓库
git remote add space https://huggingface.co/spaces/你的用户名/你的空间名

# 3. 推送代码
git add .
git commit -m "init"
git push space main
```

> ⚠️ **注意**：首次访问时会自动构建知识库（下载 embedding 模型 + 向量化 53 个 PDF），大约需要 2-5 分钟，请耐心等待。

## ⚙️ 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必填） | - |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `LLM_TEMPERATURE` | 生成温度 | `0.7` |
| `LLM_MAX_TOKENS` | 最大 token 数 | `2048` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-small-zh-v1.5` |
| `EMBEDDING_DEVICE` | 运行设备 | `cpu` |
| `CHUNK_SIZE` | 文本分块大小 | `500` |
| `CHUNK_OVERLAP` | 分块重叠 | `50` |
| `RETRIEVER_TOP_K` | 检索结果数 | `5` |

## 📁 项目结构

```
agent hr/
├── app.py                     # Gradio 主入口
├── config.py                  # 配置（环境变量读取）
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量示例
├── .gitignore                 # Git 忽略
├── README.md                  # 项目说明
├── data/
│   ├── interview_pdfs/        # 面试题 PDF（53份）
│   └── chroma_db/             # Chroma 向量库（自动生成）
├── interviews/                # 面试记录（自动生成）
└── src/
    ├── rag/
    │   ├── ingest.py          # PDF 摄取 + 向量化
    │   ├── retriever.py       # 检索器
    │   └── utils.py           # PDF 解析工具
    ├── agent/
    │   ├── interviewer.py     # 面试官 Agent
    │   ├── prompts.py         # Prompt 模板
    │   ├── state.py           # 状态管理
    │   └── evaluator.py       # 评估报告
    └── resume/
        └── parser.py          # 简历解析
```

## 📋 面试流程

### 大厂面试（3轮 × 6题）
- **第一轮**：基础知识 + 项目理解
- **第二轮**：技术深度 + 场景设计
- **第三轮**：系统设计 + 难题 + 进阶项目深挖

### 中厂面试（2轮 × 5题）
- **第一轮**：基础技术问答
- **第二轮**：项目深度挖掘

### 小厂面试（1轮 × 10题）
- 项目理解 + 深度挖掘

## 📊 评估报告

面试结束后自动生成详细报告，包含：
- 录取结论（录用/待定/不录用）
- 各轮次表现评级
- 每题详细分析（问题/回答/参考答案/评价/得分）
- 综合评价（优势/不足/建议）
- 支持 Word (.docx) 格式下载

## 🤝 部署说明

### 关于冷启动
HuggingFace Spaces 免费版在 48 小时无访问后会休眠，再次访问需要重新冷启动（约 2-5 分钟），包括：
1. 加载 embedding 模型
2. 构建向量库（首次）或加载已有向量库
3. 启动 Gradio 服务

### 优化建议
- 如果启动太慢，可以考虑升级到付费硬件
- 或者将预构建的 chroma_db 一起上传（需 git-lfs）

## 📝 License

MIT License
