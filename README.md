# Python 语法知识点问答小知识库 Agent

这是“知识点问答小知识库 Agent”课题的可运行实现。项目使用 Python、LangChain 基础组件、大模型 API 和 Streamlit，围绕 Python 语法知识点构建轻量化 RAG 问答系统。

系统支持导入知识点文档、文本分块、语义检索、混合检索、问答约束、多轮对话记忆、输入容错和来源展示。用户提问时，系统会先检索知识库片段，再要求大模型只依据库内内容回答。

## 功能概览

- 内置 Python 语法 Markdown 知识库。
- 支持上传 `.txt` / `.md` 文档并重建知识库。
- 支持简单文本分块和轻量化内存知识库。
- 支持混合检索：Embedding 语义检索、BM25、关键词匹配、编辑距离纠错。
- 支持一定程度的模糊查询，例如英文术语拼写错误、中文短语模糊匹配、部分中英文术语映射。
- 回答仅基于知识库片段；检索不到相关内容时会拒答。
- 支持多轮对话记忆，最近几轮问答会进入下一轮 Prompt。
- 支持 Streamlit 简易交互界面，展示回答内容和来源文件。

## 环境要求

- Python 3.10 或以上版本。
- Windows PowerShell、CMD、macOS Terminal 或 Linux Shell 均可运行。
- 如需真实大模型回答，需要准备 OpenAI-compatible Chat API Key，例如 DeepSeek、OpenAI 兼容服务等。
- 如需真实 Embedding 语义检索，需要准备 Embedding API Key，例如智谱 `embedding-3`、OpenAI `text-embedding-3-small` 等。

不配置 API 也可以启动项目，系统会使用本地演示模式，适合检查界面、分块和检索流程。

## 1. 获取项目

如果是从 GitHub 克隆：

```powershell
git clone https://github.com/1493002189gugecom-bit/RAG.git
cd RAG
```

如果已经在项目目录中，直接进入项目根目录即可：

```powershell
cd D:\RAG
```

## 2. 创建并激活虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可临时允许当前终端执行脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Windows CMD：

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. 安装依赖

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

项目基础依赖包括：

- `streamlit`：交互界面。
- `langchain-core`：LangChain 文档对象等基础组件。
- `langchain-text-splitters`：文本分块。
- `pytest`：测试。

如果要使用“本地 BGE Embedding”模式，还需要额外安装：

```powershell
pip install sentence-transformers
```

如果只使用默认本地混合检索 HashEmbedding 或 API Embedding，不需要安装 `sentence-transformers`。

## 4. 配置环境变量

复制示例配置文件：

```powershell
copy .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

然后编辑 `.env`。注意：`.env` 中会保存 API Key，不要提交到 GitHub。

### 4.1 DeepSeek 聊天模型配置

如果使用 DeepSeek 作为 Chat API：

```env
OPENAI_API_KEY=你的 DeepSeek API Key
OPENAI_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat
USE_FAKE_LLM=false
```

这里的 `OPENAI_API_KEY` 是聊天模型 Key，用于生成最终回答。

### 4.2 智谱 Embedding 配置

如果使用智谱 Embedding：

```env
EMBEDDING_API_KEY=你的智谱 API Key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_BACKEND=api
```

智谱常用模型：

- `embedding-3`
- `embedding-2`

### 4.3 OpenAI Embedding 配置

如果使用 OpenAI Embedding：

```env
EMBEDDING_API_KEY=你的 OpenAI API Key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BACKEND=api
```

### 4.4 离线演示配置

如果暂时没有 API Key，也可以使用离线演示模式：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1
CHAT_MODEL=deepseek-chat
USE_FAKE_LLM=true
EMBEDDING_BACKEND=hash
```

离线模式下：

- Chat 回答由 `FakeLLM` 生成，用于演示流程。
- 检索使用本地 `HashEmbedding + BM25 + 关键词 + 编辑距离`。
- 不会调用外部 API。

## 5. 启动项目

在项目根目录执行：

```powershell
streamlit run app.py
```

启动成功后，终端会显示本地访问地址，通常是：

```text
http://localhost:8501
```

在浏览器中打开该地址即可使用。

## 6. 使用流程

1. 打开 Streamlit 页面。
2. 查看顶部状态提示，确认当前是 Real API 模式还是离线演示模式。
3. 侧边栏选择 Embedding 后端：
   - `本地混合检索（HashEmbedding）`：无需 API，速度快，适合演示。
   - `本地 BGE Embedding`：需要额外安装 `sentence-transformers`，首次加载较慢。
   - `API Embedding`：调用外部 Embedding API，语义检索效果更好。
4. 可直接使用内置 Python 语法知识库，也可以上传自己的 `.txt` / `.md` 文档。
5. 点击“重建知识库”。
6. 在底部输入问题。
7. 查看回答和来源文件。

## 7. 示例问题

内置 Python 语法知识库可测试：

```text
list 和 tuple 有什么区别？
def 怎么定义函数？
字典 dict 有什么特点？
函数参数有哪些类型？
funtion 怎么定义？
```

其中 `funtion` 是故意拼错的英文词，用于测试英文编辑距离纠错。

如果上传计算机网络资料，可测试：

```text
三次握手是什么？
TCP/IP 四层模型包括哪些层？
计算机网络分几个层？
```

系统只会基于上传文档或内置知识库回答。如果知识库没有相关内容，会提示补充资料后再提问。

## 8. 模糊查询说明

本项目实现的是基础版模糊查询，主要包括：

- 英文术语编辑距离纠错：例如 `funtion` 可匹配 `function`。
- 中文 n-gram 切词：例如“三次握手”会拆成“三次”“握手”等短语参与匹配。
- BM25 和关键词召回：提高精确词命中的稳定性。
- 部分中英文术语扩展：例如“列表”扩展到 `list`，“元组”扩展到 `tuple`，“函数”扩展到 `function` / `def`。
- Embedding 语义检索：使用 API Embedding 或本地 Embedding 时，可以提升语义相似问题的召回效果。

注意：这不是完整搜索引擎，也不是专业中文纠错模型。如果知识库只有英文术语，而用户用中文提问，需要依赖 Embedding 语义能力或手写术语映射。

## 9. 多轮对话记忆说明

`rag_agent/memory.py` 会保存最近多轮问答。下一轮提问时，Agent 会把历史对话放入 Prompt，让模型理解上下文。

可以这样测试：

```text
第一问：三次握手是什么？
第二问：第二次是谁发的？
第三问：为什么还要第三次？
```

如果知识库中有三次握手相关内容，后两问可以借助历史上下文理解“第二次”“第三次”的指代。

## 10. 运行测试

在项目根目录执行：

```powershell
pytest -q
```

测试覆盖内容包括：

- 文档加载。
- 文本分块。
- Embedding 封装。
- 检索器。
- RAG Agent 问答约束。
- 多轮对话记忆。
- API 配置读取。

## 11. 常见问题

### 11.1 页面能启动，但回答像模板

可能当前是离线演示模式。检查 `.env`：

```env
USE_FAKE_LLM=false
OPENAI_API_KEY=你的真实 Chat API Key
```

然后重启 Streamlit。

### 11.2 使用 API Embedding 建库失败

可能原因：

- `EMBEDDING_API_KEY` 填错。
- `EMBEDDING_BASE_URL` 填错。
- 网络超时或 SSL 连接中断。
- 上传文档过大。

可先切回本地混合检索模式，确认文档内容和问答流程正常。

### 11.3 本地 BGE Embedding 加载慢

首次使用会下载或加载模型，这是正常现象。如果只是课堂演示，建议使用默认 HashEmbedding 或 API Embedding。

### 11.4 上传文档后问不到内容

建议检查：

- 上传文件是否是 `.txt` 或 `.md`。
- 文件编码是否为 UTF-8。
- 是否点击了“重建知识库”。
- 问题是否确实在文档内容范围内。
- 是否选择了合适的 Embedding 后端。

## 12. 项目结构

```text
RAG/
  app.py                         # Streamlit 交互入口
  requirements.txt               # Python 依赖
  .env.example                   # 环境变量示例
  README.md                      # 项目说明与安装教程
  data/
    knowledge/
      python_syntax.md           # 内置 Python 语法知识库
  docs/
    project_plan.md              # 项目计划文档
    internship_report_template.md# 实习报告模板
  rag_agent/
    agent.py                     # RAG 问答流程
    config.py                    # 环境配置读取
    documents.py                 # 文档加载
    embeddings.py                # Embedding 封装
    llm.py                       # 大模型 API 封装
    memory.py                    # 多轮对话记忆
    retriever.py                 # 内存知识库与混合检索
    splitting.py                 # 文本分块
  tests/                         # 自动化测试
```

## 13. 交付说明

本项目满足课题要求：

- 技术栈：Python + 大模型 API + LangChain 基础组件 + Streamlit。
- 基础功能：多轮对话记忆、输入容错、模块化代码结构。
- 知识库能力：导入文档、文本分块、轻量化知识库、语义检索、RAG 问答约束。
- 模糊查询：混合检索、编辑距离纠错、关键词匹配和基础中英文术语扩展。
- 交付内容：可运行源码、项目计划文档、实习报告模板、测试用例。
