# 知识点问答小知识库 Agent

基于 RAG 的轻量化知识库问答系统。上传课程文档，系统自动切分、建索引，用户提问时检索相关片段再由大模型基于片段回答。

## 环境要求

- **Python 3.10 或以上版本**（可在终端用 `python --version` 检查）
- **Windows PowerShell / CMD** 或 **macOS / Linux Terminal**
- 如需真实大模型回答，需要 OpenAI-compatible Chat API Key（如 DeepSeek、OpenAI）

不配 API 也可以启动，默认使用离线演示模式。

## 安装教程

### 第一步：确认 Python 版本

```powershell
python --version
```

确保输出 `Python 3.10.x` 或更高版本。如果版本低于 3.10，请先安装或升级 Python。

### 第二步：创建虚拟环境

**Windows PowerShell：**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果提示"无法加载文件...因为在此系统上禁止运行脚本"，先执行：
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Windows CMD：**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux：**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

激活后终端前面会出现 `(.venv)` 标识。

### 第三步：安装依赖

```powershell
pip install -r requirements.txt
```

这会在虚拟环境中安装核心依赖（streamlit、langchain-core 等）。

### 第四步：（可选）安装本地语义模型

如需使用"本地 BGE Embedding"模式（中文语义检索效果更好），还需要：
```powershell
pip install sentence-transformers
```

首次使用时系统会自动下载模型文件（约 95MB），下载一次后续离线使用。如果下载慢，可在 `.env` 中配置镜像：
```env
HF_ENDPOINT=https://hf-mirror.com
```

### 第五步：启动

```powershell
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。默认使用离线模式，无需 API Key。

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

## 使用教程

### 第一次使用

```
1. 启动项目（streamlit run app.py）
   → 浏览器自动打开 http://localhost:8501

2. 选择 Embedding 后端（侧边栏）
   ┌─────────────────────────────────────┐
   │ ○ 本地混合检索（HashEmbedding）       │ ← 默认，零依赖，速度快
   │ ○ 本地 BGE Embedding                │ ← 中文语义最好，需安装模型
   │ ○ API Embedding                     │ ← 效果最好，需 API Key
   └─────────────────────────────────────┘

3. 使用内置知识库或上传自己的文档
   → 默认已加载 Python 语法知识库
   → 如需用自己的资料，点"上传知识库文档"上传 .txt/.md 文件

4. 点击"重建知识库"
   → 系统自动切分文档 → 向量化 → 建索引
   → 成功后显示"共 N 个知识块"

5. 在底部输入框提问
   例如：list 和 tuple 有什么区别？
   → 系统检索相关片段 → LLM 基于片段回答
   → 下方可展开"📚 检索依据"看命中了哪些片段
```

### 常用操作

| 操作 | 位置 | 说明 |
|------|------|------|
| 新建对话 | 侧边栏 → 💬 对话历史 → ✏️ 新建 | 清空当前对话，开始新一轮 |
| 加载历史 | 侧边栏 → 💬 对话历史 → 点击对话标题 | 恢复之前的问答记录 |
| 导出笔记 | 聊天区底部 → 📥 导出全部笔记 | 全部问答导出为 Markdown，可直接导入 Obsidian |
| 清空对话 | 侧边栏 → 清空对话 | 只清空聊天记录，不重建知识库 |
| 切换后端 | 侧边栏 → Embedding 后端 | 切换后必须点"重建知识库"才能生效 |

### 示例问题

内置 Python 知识库可以测试：
```text
列表和元组有什么区别？
def 怎么定义函数？
字典 dict 有什么特点？
函数参数有哪些类型？
funtion 怎么定义？
```

> `funtion` 是故意拼错的，用于测试编辑距离纠错（会自动纠正为 `function`）。

### 问题答不出来怎么办

1. 确认知识库里确实有相关内容——"重建知识库"后展开"📄 知识块详情"查看
2. 切换 Embedding 后端后记得重建知识库
3. 检查上传文件编码是否为 UTF-8
4. 如果"检索依据"为空，说明检索没找到，试试换一种说法问

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
