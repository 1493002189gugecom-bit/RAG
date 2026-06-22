# Python 语法知识库 Agent

这是“题目5：知识点问答小知识库 Agent”的可运行实现。项目使用 Python、LangChain 基础组件、OpenAI 兼容大模型 API 和 Streamlit，围绕 Python 语法知识点构建轻量 RAG 问答系统。

## 功能
- 内置 Python 语法 Markdown 知识库。
- 支持上传 `.txt` / `.md` 文档重建知识库。
- 支持文本分块、语义检索、模糊匹配。
- 回答仅基于知识库片段，检索不到时拒答。
- 支持多轮对话记忆和来源片段展示。
- 默认可离线演示：使用 `HashEmbedding` 和 `FakeLLM`。

## 安装
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置
复制 `.env.example` 为 `.env`，或直接设置环境变量。项目使用 OpenAI-compatible API 格式，因此 DeepSeek、OpenAI 兼容代理、本地兼容服务都可以接入。

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"
$env:CHAT_MODEL="deepseek-chat"
$env:EMBEDDING_API_KEY="你的 Embedding API Key"
$env:EMBEDDING_BASE_URL="https://api.openai.com/v1"
$env:EMBEDDING_MODEL="text-embedding-3-small"
```

如果不配置 `OPENAI_API_KEY`，界面会自动使用测试模式回答，适合课堂演示。真实 API 模式下，聊天回答会调用 `CHAT_MODEL`。

检索层现在是混合检索：向量相似度 + BM25 + 关键词匹配 + 编辑距离纠错。默认使用本地 HashEmbedding 参与向量分数；如果你配置了 `EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL`，可以在侧边栏勾选 “Use API Embedding” 来使用真实 Embedding API。DeepSeek Chat API 不一定提供 Embedding endpoint，因此 Chat 和 Embedding 配置是分开的。

Embedding API 请求支持分批和自动重试，遇到临时 SSL 断连时会重试；如果外部接口持续失败，可以取消勾选 “Use API Embedding”，系统仍会使用本地混合检索。

常见配置：

```env
# DeepSeek
OPENAI_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat

# OpenAI
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
EMBEDDING_API_KEY=你的 OpenAI API Key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# Zhipu Embedding
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
```

当前项目已验证智谱 `embedding-3`：返回 2048 维向量，可在侧边栏勾选 “Use API Embedding” 后用于上传文档的语义检索。

## 运行
```powershell
streamlit run app.py
```

当前本机已按 DeepSeek OpenAI-compatible API 配置 `.env`。打开页面后，如果顶部显示 Real API mode，说明聊天回答会调用真实 DeepSeek API。

## 测试
```powershell
pytest -q
```

## 已验证的真实 API 问题
- `list 和 tuple 有什么区别？`
- `def 怎么定义函数？`
- `怎么维修电脑硬件？`

前两个问题会基于 Python 语法知识库回答，第三个问题不在知识库范围内，会触发拒答。

也已验证上传计算机网络类文档后：
- `三次握手`
- `计算机网络分几个层？`

本地混合检索和智谱 Embedding 模式均可召回相关片段。

## 目录结构
```text
rag_agent/
  agent.py        # RAG 问答流程
  config.py       # 环境配置
  documents.py    # 文档加载
  embeddings.py   # Embedding 封装
  llm.py          # 大模型 API 封装
  memory.py       # 多轮记忆
  retriever.py    # 内存向量检索
  splitting.py    # 文本分块
data/knowledge/
  python_syntax.md
docs/
  project_plan.md
  internship_report_template.md
tests/
```
