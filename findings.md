# Findings

## Project Context
- 工作目录：`D:\RAG`。
- 初始状态：目录为空，不是 Git 仓库。
- 用户要求：先生成项目计划文档，然后开始搭建题目5“知识点问答小知识库 Agent”。
- 用户最终选择课程主题：Python 语法。

## Requirement Interpretation
- 技术栈必须包含 Python、大模型 API、LangChain 基础组件、Gradio/Streamlit 简易交互界面。
- 基础标配功能包括多轮对话记忆、输入容错、模块化代码结构。
- 功能主题为轻量 RAG：知识点文档导入、文本分块、语义检索、只基于库内内容作答、支持模糊查询。

## Design Notes
- 内置知识库使用 `data/knowledge/python_syntax.md`。
- Python 语法知识点覆盖变量、数据类型、列表、元组、字典、条件、循环、函数、异常处理、模块导入。
- Streamlit 适合本项目：单文件 UI 即可覆盖上传、对话、引用展示和配置提示。
- 为了降低安装复杂度，核心检索先使用 LangChain 文档对象 + 自定义内存向量检索；后续可替换为 FAISS/Chroma。
- 测试中不调用真实外部 API，通过 Fake Embedding 和 Fake LLM 验证流程。

## Verification Strategy
- 单元测试覆盖：
  - 空输入容错。
  - 文档加载和分块。
  - 语义检索和模糊检索。
  - “无库内依据则拒答”的问答约束。
  - 多轮记忆窗口。
- 手动验证覆盖：
  - `streamlit run app.py` 能启动界面。
  - 上传或使用内置知识库后可以问答并展示来源片段。
