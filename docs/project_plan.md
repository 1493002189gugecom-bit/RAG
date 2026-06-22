# 知识点问答小知识库 Agent 项目计划文档

## 1. 项目名称
知识点问答小知识库 Agent

## 2. 选题方向
专业基础课选择：Python 语法。

系统内置一份 Python 语法知识点 Markdown 文档，内容覆盖变量、数据类型、列表、元组、字典、条件语句、循环、函数、异常处理、模块导入等基础知识。学生也可以替换或上传其他课程的 `.txt` / `.md` 文档来构建新的小型知识库。

## 3. 建设目标
搭建一个轻量化 RAG 问答系统，让用户围绕知识库文档提问。系统先对文档进行文本分块，再将分块转换为向量，用户提问时通过语义检索和模糊匹配召回相关片段，最后调用大模型 API 生成答案。答案必须受知识库约束：没有检索到依据时，不编造内容，而是提示“知识库中未找到相关内容”。

## 4. 技术栈
- 开发语言：Python
- 大模型接口：OpenAI-compatible Chat API，例如 DeepSeek 或 OpenAI
- Embedding 接口：OpenAI-compatible Embedding API，可与 Chat API 分开配置；默认演示使用本地 Hash Embedding，测试环境使用 Fake Embedding
- LangChain 基础组件：`Document`、文本分块器、Embedding 抽象
- 交互界面：Streamlit
- 测试框架：pytest

## 5. 核心功能
- 知识点文档导入：支持项目内置 Markdown，也支持界面上传 `.txt` / `.md`。
- 简单文本分块：按固定长度和重叠窗口切分知识点内容。
- 混合检索：融合向量相似度、BM25、关键词匹配和编辑距离纠错，提升召回稳定性。
- 模糊查询：当用户提问词语不完全一致或英文拼写错误时，基于当前知识库自动提取词表并纠错后补充召回。
- 约束问答：只基于召回片段生成答案，无法命中时拒答。
- 多轮对话记忆：保存最近若干轮问答，作为大模型回答时的上下文参考。
- 输入容错：处理空问题、未构建知识库、文档为空、API 配置缺失等情况。

## 6. 模块划分
- `rag_agent/config.py`：读取环境变量和默认配置。
- `rag_agent/documents.py`：加载内置文档和上传文档。
- `rag_agent/splitting.py`：封装文本分块逻辑。
- `rag_agent/retriever.py`：实现内存向量库、语义检索和模糊检索。
- `rag_agent/memory.py`：管理多轮对话记忆。
- `rag_agent/llm.py`：封装真实 LLM 调用和测试用 Fake LLM。
- `rag_agent/agent.py`：组织 RAG 流程，输出答案和来源。
- `app.py`：Streamlit 界面入口。

## 7. 五个工作日安排
| 时间 | 任务 | 产出 |
|---|---|---|
| 第 1 天 | 需求分析、项目计划、模块设计、测试设计 | 项目计划文档、目录结构、核心测试 |
| 第 2 天 | 文档加载、文本分块、检索模块 | 可测试的知识库构建与检索功能 |
| 第 3 天 | RAG Agent、大模型 API 封装、多轮记忆 | 可命令行/测试调用的问答流程 |
| 第 4 天 | Streamlit 页面、上传文档、问答展示 | 可运行交互界面 |
| 第 5 天 | 测试、说明文档、实习报告整理 | README、测试结果、实习报告模板 |

## 8. 测试方案
- 使用 pytest 做核心模块单元测试。
- 使用 Fake Embedding 验证检索排序，不依赖真实 API。
- 使用 Fake LLM 验证 Prompt 中确实包含检索片段和历史对话。
- 对空输入、空知识库、检索无结果等异常情况做容错测试。
- 最后运行 `pytest` 和基础导入检查，确认源码可运行。

## 9. 交付清单
- 可运行源码。
- `docs/project_plan.md`：项目计划文档。
- `docs/internship_report_template.md`：实习报告模板，不粘贴源码，包含设计实现思路、测试与分工。
- `README.md`：安装、配置、运行和测试说明。
- `.env.example`：API 配置示例。
- `data/knowledge/python_syntax.md`：样例知识库。
