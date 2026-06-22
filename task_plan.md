# 知识点问答小知识库 Agent 项目计划

## Goal
在 `D:\RAG` 从零搭建一个可运行的“题目5：知识点问答小知识库 Agent”，主题选择 Python 语法，满足 Python + 大模型 API + LangChain 基础组件 + Streamlit 简易界面，并具备多轮对话记忆、输入容错、模块化代码结构。

## Scope
- 课程主题：Python 语法。
- 知识库来源：项目内置 Markdown 知识点文档，也支持用户在界面上传 `.txt` / `.md`。
- 回答约束：只允许基于检索到的库内内容回答；检索不到时明确说明知识库未覆盖。
- 检索能力：文本分块、Embedding 语义检索、简单模糊匹配补充召回。
- 交付物：可运行源码、项目计划文档、实习报告模板、测试用例。

## Phases

### Phase 1: 项目计划与交付结构
Status: complete
- 创建持久规划文件：`task_plan.md`、`findings.md`、`progress.md`。
- 创建中文项目计划文档：`docs/project_plan.md`。
- 确定 5 个工作日开发安排、模块划分、测试策略和交付清单。

### Phase 2: 测试优先的核心模块
Status: complete
- 先写核心模块测试：文本分块、知识库检索、问答约束、对话记忆。
- 运行测试并确认因模块未实现而失败。
- 实现最小核心代码让测试通过。

### Phase 3: LangChain 与大模型 API 接入
Status: complete
- 接入 LangChain `Document`、文本分块组件、Embedding 接口。
- 封装 OpenAI 兼容 Chat API 和 Embedding API 配置。
- 保持测试可用的 Fake Embedding / Fake LLM 注入点。

### Phase 4: Streamlit 交互界面
Status: complete
- 实现上传知识文档、构建知识库、提问、展示引用片段、多轮上下文。
- 增加输入容错和错误提示。

### Phase 5: 文档、样例数据与验证
Status: complete
- 创建 Python 语法样例知识库文档。
- 编写 README、`.env.example`、实习报告模板。
- 运行测试并做基础导入验证。

## Decisions
- UI 选择 Streamlit：代码量小，适合 5 个工作日实习项目交付。
- 向量库选择轻量内存实现：减少 FAISS 安装门槛，同时保留 Embedding 语义检索流程。
- 课程方向选择 Python 语法：贴近入门教学，适合展示概念解释、语法规则、示例说明和模糊查询。
- LLM/Embedding 通过环境变量配置，兼容 OpenAI 及同类 API。

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| 当前目录不是 Git 仓库 | `git status --short` | 不执行提交，改为直接创建项目文件并在最终说明中注明 |
| 核心模块不存在 | `pytest -q` | 预期 TDD 红灯，已创建 `rag_agent` 包 |
| 文档与 LLM 模块不存在 | `pytest -q` | 预期 TDD 红灯，接下来创建 `documents.py` 和 `llm.py` |
| Embedding 模块不存在 | `pytest -q` | 预期 TDD 红灯，接下来创建 `embeddings.py` |
| 编译检查通配符未展开 | `python -m py_compile app.py rag_agent\*.py` | 改用 PowerShell 遍历文件后编译 |
