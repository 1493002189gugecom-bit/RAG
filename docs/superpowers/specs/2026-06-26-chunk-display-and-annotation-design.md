# 分块结果展示与查询标注设计

## 背景

用户重建知识库后看不到文本分块的具体结果，问答时也不知道检索命中了哪些片段、命中在什么位置。需要将分块信息和检索命中可视化，直接展示在聊天主界面。

## 改动范围

3 个文件，约 50 行新增代码：

| 文件 | 改动 |
|------|------|
| `rag_agent/agent.py` | `QAResponse` 增加 `scores` 字段 |
| `rag_agent/retriever.py` | 新增 `highlight_terms()` 工具函数 |
| `app.py` | 存储 chunks + 展示分块列表 + 展示检索标注 |

不涉及检索逻辑、splitting、API 层的修改。

## 1. QAResponse 增加 scores 字段

**`rag_agent/agent.py`**

```python
@dataclass(frozen=True)
class QAResponse:
    answer: str
    sources: list[str]
    contexts: list[str]
    scores: list[float] | None = None  # 新增
```

`answer()` 方法中返回时带上得分：

```python
scores = [result.score for result in results]
return QAResponse(
    answer=answer,
    sources=sources,
    contexts=contexts,
    scores=scores,           # 新增
)
```

## 2. 高亮工具函数

**`rag_agent/retriever.py`** — 新增模块级函数：

```python
def highlight_terms(text: str, query: str) -> str:
    """将 query 中的关键词在 text 中用 **粗体** 标记。"""
    tokens = _tokenize(query)
    for token in sorted(set(tokens), key=len, reverse=True):
        if _is_indexable_token(token):
            text = text.replace(token, f"**{token}**")
    return text
```

按词长降序替换，避免短词破坏长词匹配。复用已有的 `_tokenize()` 和 `_is_indexable_token()`。

## 3. UI 展示

### 3.1 重建知识库后展示分块列表

**`app.py`** — 重建按钮处理逻辑中：

- 将 `chunks` 存到 `st.session_state.chunks`
- 在 `st.success()` 后新增 `st.expander("📄 知识块详情")`
- 每个块显示：序号、来源文件名、字数、内容前 200 字

### 3.2 问答结果中标注命中位置

**`app.py`** — 助手回答渲染区：

- `st.markdown(response.answer)` 之后新增 `st.expander("📚 检索依据")`
- 每个片段显示：序号、来源文件、检索得分、高亮文本（前 300 字）
- 高亮通过 `highlight_terms(chunk_text, question)` 实现

## 数据流

```
重建知识库:
  上传/内置文档 → split_documents → chunks
    → 存入 session_state.chunks（新增）
    → 展示折叠面板

问答:
  用户提问 → agent.answer() 返回 QAResponse（含 scores）
    → 展示回答
    → 展示折叠面抜 "检索依据"
      → highlight_terms() 标注匹配词
```

## 不涉及

- spliting.py 不改
- retriever.py 检索逻辑不改
- 测试不需要改（QAResponse 新增可选字段向后兼容）
