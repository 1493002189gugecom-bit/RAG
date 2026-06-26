# 知识点问答小知识库 Agent

基于 RAG 的轻量化知识库问答系统。上传课程文档，系统自动切分、建索引，用户提问时检索相关片段再由大模型基于片段回答。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。默认使用离线演示模式，无需 API Key。

## 功能

| 功能 | 说明 |
|------|------|
| 内置知识库 | 内置 Python 语法文档，开箱即用 |
| 上传文档 | 支持 `.txt` / `.md`，上传后重建知识库 |
| 文本分块 | 自动切分为 280 字片段，相邻重叠 40 字 |
| 混合检索 | 语义向量 + BM25 + 关键词 + 编辑距离 加权排序 |
| 模糊查询 | 英文拼写纠错（`funtion` → `function`）、CJK n-gram、中英文术语映射 |
| LLM 消解 | 检索无结果时自动用 LLM 改写查询重试 |
| 指代消解 | "两者" → "列表、元组"，"它们" → 前文主题 |
| 多轮记忆 | 保留最近 4 轮对话，支持上下文追问 |
| 知识块可视化 | 折叠面板展示分块详情，问答标注命中位置 |
| 对话持久化 | 自动保存，重启后可继续之前的对话 |
| 导出笔记 | 全部问答导出为 Markdown 笔记文件 |
| 回答约束 | 只基于知识库片段回答，检索不到拒答 |

## 三种 Embedding 方式

| 方式 | 效果 | 依赖 |
|------|------|------|
| 本地混合检索（HashEmbedding） | 精确匹配，速度最快 | 零依赖 |
| 本地 BGE Embedding | 中文语义检索 | `pip install sentence-transformers` |
| API Embedding | 效果最好 | API Key + 网络 |

侧边栏可随时切换 Embedding 后端，切换后点击"重建知识库"生效。

## 配置

复制 `.env.example` 为 `.env`，按需配置：

**Chat API（回答用）：**
```env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat
```

**Embedding API（检索用，可选）：**
```env
EMBEDDING_API_KEY=xxx
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_BACKEND=api
```

**不配 API 也能用：** 默认使用 `FakeLLM` + `HashEmbedding`，适合演示和测试。

## 使用流程

1. 启动后侧边栏选择 Embedding 后端
2. 上传 `.txt` / `.md` 文档（或使用内置 Python 知识库）
3. 点击"重建知识库"
4. 底部输入问题

## 测试

```powershell
pytest -q
```

覆盖：文档加载、分块、检索、纠错、Agent 流程、多轮记忆。

## 项目结构

```
RAG/
  app.py                         # Streamlit 交互入口
  rag_agent/
    agent.py                     # RAG 问答编排（检索→Prompt→回答）
    config.py                    # 环境变量读取
    conversation_store.py        # 对话持久化存储
    documents.py                 # 文档加载
    embeddings.py                # Embedding（Hash / BGE / API）
    llm.py                       # Chat API 封装
    memory.py                    # 多轮对话记忆
    retriever.py                 # 混合检索（语义 + BM25 + 关键词 + 编辑距离）
    splitting.py                 # 文本分块
  data/knowledge/                # 内置知识库
  conversations/                 # 保存的对话（自动生成）
  tests/                         # 自动化测试
```

## 依赖

- Python ≥ 3.10
- `streamlit` — 交互界面
- `langchain-core` — 文档对象
- `langchain-text-splitters` — 文本分块
- `sentence-transformers`（可选）— 本地 BGE Embedding

详细依赖见 `requirements.txt`。
