from __future__ import annotations

from datetime import datetime

from langchain_core.documents import Document

import streamlit as st

from rag_agent.agent import KnowledgeQAAgent, QAResponse
from rag_agent.config import load_config
from rag_agent.conversation_store import ConversationStore
from rag_agent.documents import load_builtin_documents
from rag_agent.embeddings import (
    HashEmbedding,
    OpenAICompatibleEmbedding,
)
from rag_agent.llm import FakeLLM, OpenAICompatibleLLM
from rag_agent.memory import ConversationMemory
from rag_agent.retriever import InMemoryKnowledgeBase, highlight_terms
from rag_agent.splitting import split_documents


def _export_conversation(messages: list[dict], title: str) -> str:
    """Format conversation as a clean Markdown study note."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_sources: list[str] = []
    lines = [
        f"# {title}",
        "",
        f"> 导出时间：{now}",
        f"> 知识点数：{len([m for m in messages if m['role'] == 'user'])}",
        "",
        "---",
        "",
    ]
    pairs = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            pairs.append({"q": msg["content"], "a": "", "sources": [], "contexts": []})
        elif msg["role"] == "assistant" and pairs:
            pairs[-1]["a"] = msg["content"]
            pairs[-1]["sources"] = msg.get("sources", [])
            pairs[-1]["contexts"] = msg.get("contexts", [])

    for i, pair in enumerate(pairs, 1):
        lines.append(f"## {i}. {pair['q']}")
        lines.append("")
        lines.append(pair["a"])
        lines.append("")
        if pair["sources"]:
            src_str = "，".join(pair["sources"])
            lines.append(f"> **来源：** {src_str}")
            all_sources.extend(s for s in pair["sources"] if s not in all_sources)
        if pair["contexts"]:
            for j, ctx in enumerate(pair["contexts"], 1):
                lines.append(f"> **片段 #{j}** {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        lines.append("")
        lines.append("---")
        lines.append("")

    if all_sources:
        sources_section = "\n".join(f"- {s}" for s in sorted(set(all_sources)))
        lines.append(f"## 参考来源\n\n{sources_section}\n")

    return "\n".join(lines)


st.set_page_config(page_title="知识点问答小知识库 Agent", page_icon="📚")


def build_documents_from_uploads(uploaded_files) -> list[Document]:
    documents: list[Document] = []
    max_file_size = 5 * 1024 * 1024  # 5MB
    for uploaded_file in uploaded_files:
        # 检查文件扩展名，非 .txt / .md 的跳过
        filename = uploaded_file.name.lower()
        if not (filename.endswith(".txt") or filename.endswith(".md")):
            st.warning(f"文件 {uploaded_file.name} 格式不支持，仅支持 .txt 和 .md 文件，已跳过。")
            continue
        file_size = len(uploaded_file.getvalue())
        if file_size > max_file_size:
            st.warning(f"文件 {uploaded_file.name} 超过 5MB 大小限制，已跳过。")
            continue
        if file_size == 0:
            st.warning(f"文件 {uploaded_file.name} 为空，已跳过。")
            continue
        try:
            text = uploaded_file.getvalue().decode("utf-8").strip()
        except UnicodeDecodeError:
            st.warning(
                f"文件 {uploaded_file.name} 编码不是 UTF-8，无法读取。"
                "请保存为 UTF-8 编码后重试。"
            )
            continue
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name},
                )
            )
    return documents


@st.cache_resource(show_spinner="正在加载本地 Embedding 模型，首次启动可能需要几十秒...")
def get_local_embedding(model_name: str, device: str):
    from rag_agent.embeddings import LocalSentenceTransformerEmbedding  # 延迟导入

    embedding = LocalSentenceTransformerEmbedding(model_name=model_name, device=device)
    embedding.preload()
    return embedding


def create_embedding(backend: str):
    config = _get_config()
    if backend == "api":
        return OpenAICompatibleEmbedding(
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
        )
    if backend == "local":
        return get_local_embedding(
            config.local_embedding_model,
            config.local_embedding_device,
        )
    return HashEmbedding()


def create_knowledge_base(documents: list[Document], embedding_backend: str):
    embedding = create_embedding(embedding_backend)
    chunks = split_documents(documents, chunk_size=280, chunk_overlap=40)
    knowledge_base = InMemoryKnowledgeBase(embedding_model=embedding)
    knowledge_base.add_documents(chunks)
    return knowledge_base, chunks


def create_llm():
    config = _get_config()
    if config.use_fake_llm or not config.openai_api_key:
        return FakeLLM()
    return OpenAICompatibleLLM(
        api_key=config.openai_api_key,
        model=config.chat_model,
        base_url=config.openai_base_url,
    )


def initialize_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=4)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    if "conv_store" not in st.session_state:
        st.session_state.conv_store = ConversationStore()
    if "knowledge_base" not in st.session_state:
        config = load_config()
        docs = load_builtin_documents()
        backend = "local" if config.preload_local_embedding else "hash"
        kb, chunks = create_knowledge_base(docs, embedding_backend=backend)
        st.session_state.knowledge_base = kb
        st.session_state.chunk_count = len(chunks)
        st.session_state.embedding_backend = backend


@st.cache_resource
def _get_config():
    return load_config()


initialize_state()
config = _get_config()

st.title("知识点问答小知识库 Agent")
st.caption("基于 RAG 的课程知识库问答系统：先检索知识片段，再由大模型基于片段回答。")

if config.api_ready:
    st.success(
        "已启用真实 Chat API。检索采用混合检索：向量相似度 + BM25 + 关键词匹配 + 编辑距离纠错。"
    )
elif not config.openai_api_key:
    st.info(
        "未配置 OPENAI_API_KEY，当前为离线演示模式：使用 HashEmbedding + FakeLLM，"
        "可测试界面、分块和检索流程。"
    )
else:
    st.warning(
        "已检测到 API Key，但 USE_FAKE_LLM=true 或模型/Base URL 配置不完整，"
        "当前仍使用离线演示模式。"
    )

with st.sidebar:
    st.subheader("知识库设置")
    st.write(f"当前知识块数量：{st.session_state.get('chunk_count', 0)}")
    backend_options = {
        "本地混合检索（HashEmbedding）": "hash",
        "本地 BGE Embedding": "local",
        "API Embedding": "api",
    }
    default_backend = config.embedding_backend
    default_label = next(
        (label for label, value in backend_options.items() if value == default_backend),
        "本地混合检索（HashEmbedding）",
    )
    selected_backend_label = st.radio(
        "Embedding 后端",
        list(backend_options.keys()),
        index=list(backend_options.keys()).index(default_label),
        help="本地 BGE Embedding 首次加载模型较慢，加载后会缓存复用；API Embedding 调用外部向量接口。",
    )
    embedding_backend = backend_options[selected_backend_label]
    if embedding_backend == "api" and not config.embedding_api_ready:
        st.caption(
            "尚未配置 API Embedding。系统仍会使用本地 HashEmbedding + BM25 + "
            "关键词匹配 + 编辑距离纠错进行混合检索。"
        )
        embedding_backend = "hash"
    if embedding_backend == "local":
        st.caption(
            f"本地模型：{config.local_embedding_model}，设备：{config.local_embedding_device}。"
            "首次加载后会缓存，后续重建知识库会复用。"
        )
    uploaded_files = st.file_uploader(
        "上传知识库文档（支持 .txt / .md，其他格式会被自动忽略）",
        accept_multiple_files=True,
    )
    if st.button("重建知识库", use_container_width=True):
        try:
            documents = (
                build_documents_from_uploads(uploaded_files)
                if uploaded_files
                else load_builtin_documents()
            )
            if not documents:
                st.warning("文档内容为空，请上传有效的 .txt 或 .md 文件。")
            else:
                kb, chunks = create_knowledge_base(documents, embedding_backend)
                st.session_state.knowledge_base = kb
                st.session_state.chunks = chunks
                st.session_state.chunk_count = len(chunks)
                st.session_state.embedding_backend = embedding_backend
                st.session_state.memory = ConversationMemory(max_turns=4)
                st.session_state.messages = []
                st.session_state.current_conv_id = None
                mode = {
                    "api": "API Embedding",
                    "local": "本地 BGE Embedding",
                    "hash": "本地混合检索",
                }[embedding_backend]
                st.success(f"知识库已重建：共 {len(chunks)} 个知识块，检索模式：{mode}。")
        except Exception as exc:
            st.error(f"未能建立知识库：{exc}")

    if st.button("清空对话", use_container_width=True):
        st.session_state.memory = ConversationMemory(max_turns=4)
        st.session_state.messages = []
        st.session_state.current_conv_id = None
        st.rerun()

    # ── 对话历史 ──────────────────────────────────────────
    st.divider()
    st.subheader("💬 对话历史")
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.caption("会话列表")
    with col2:
        if st.button("✏️ 新建", use_container_width=True):
            st.session_state.memory = ConversationMemory(max_turns=4)
            st.session_state.messages = []
            st.session_state.current_conv_id = None
            st.rerun()

    conversations = st.session_state.conv_store.list()
    if conversations:
        for conv in conversations:
            cols = st.columns([0.8, 0.2])
            with cols[0]:
                is_current = conv["id"] == st.session_state.current_conv_id
                label = conv["title"][:18]
                if len(conv["title"]) > 18:
                    label += "…"
                if is_current:
                    st.markdown(f"**▶ {label}**")
                else:
                    if st.button(label, key=f"load_{conv['id']}", use_container_width=True):
                        data = st.session_state.conv_store.load(conv["id"])
                        if data:
                            st.session_state.messages = data["messages"]
                            memory = ConversationMemory(max_turns=4)
                            for q, a in data["turns"]:
                                memory.add_turn(q, a)
                            st.session_state.memory = memory
                            st.session_state.current_conv_id = conv["id"]
                            st.rerun()
            with cols[1]:
                if st.button("✕", key=f"del_{conv['id']}"):
                    st.session_state.conv_store.delete(conv["id"])
                    if conv["id"] == st.session_state.current_conv_id:
                        st.session_state.memory = ConversationMemory(max_turns=4)
                        st.session_state.messages = []
                        st.session_state.current_conv_id = None
                    st.rerun()
    else:
        st.caption("暂无对话记录")

# ── 主内容区 ─────────────────────────────────────────────

# 知识块概览（展开显示当前分块详情）
chunks = st.session_state.get("chunks")
if chunks is not None and len(chunks) > 0:
    with st.expander(f"📄 知识块详情（共 {len(chunks)} 个片段）", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "未知来源")
            text = chunk.page_content.strip()
            st.markdown(f"**#{i}** `{source}`（{len(text)} 字）")
            st.markdown(f"> {text[:200]}{'...' if len(text) > 200 else ''}")
            if i < len(chunks):
                st.divider()

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("来源：" + "，".join(message["sources"]))
        # 历史对话的检索依据标注
        if message.get("role") == "assistant" and message.get("contexts"):
            contexts = message["contexts"]
            scores = message.get("scores") or []
            with st.expander(f"📚 检索依据（{len(contexts)} 个片段）", expanded=False):
                user_q = st.session_state.messages[
                    st.session_state.messages.index(message) - 1
                ]["content"]
                for i, (ctx, src) in enumerate(zip(contexts, message["sources"]), 1):
                    score = scores[i - 1] if i <= len(scores) else 0
                    highlighted = highlight_terms(ctx, user_q)
                    tail = "..." if len(highlighted) > 300 else ""
                    st.markdown(f"**#{i}** `{src}` 得分 **{score:.2f}**")
                    st.markdown(f"{highlighted[:300]}{tail}")
                    if i < len(contexts):
                        st.divider()

question = st.chat_input("请输入问题，例如：三次握手是什么？ / list 和 tuple 有什么区别？")
if question is not None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    agent = KnowledgeQAAgent(
        st.session_state.knowledge_base,
        create_llm(),
        st.session_state.memory,
    )
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                response = agent.answer(question)
            except Exception as exc:
                response = QAResponse(
                    answer=f"处理问题时出错：{exc}",
                    sources=[],
                    contexts=[],
                )
        st.markdown(response.answer)
        if response.sources:
            st.caption("来源：" + "，".join(response.sources))

        # 展示检索依据（命中片段 + 关键词高亮）
        if response.contexts:
            with st.expander(f"📚 检索依据（{len(response.contexts)} 个片段）", expanded=False):
                for i, (ctx, src) in enumerate(zip(response.contexts, response.sources), 1):
                    score = response.scores[i - 1] if response.scores else 0
                    highlighted = highlight_terms(ctx, question)
                    tail = "..." if len(highlighted) > 300 else ""
                    st.markdown(f"**#{i}** `{src}` 得分 **{score:.2f}**")
                    st.markdown(f"{highlighted[:300]}{tail}")
                    if i < len(response.contexts):
                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "contexts": response.contexts,
            "scores": response.scores,
        }
    )

    # 每次助手回复后自动保存对话
    if st.session_state.current_conv_id is None:
        st.session_state.current_conv_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    title = st.session_state.messages[0]["content"][:30]
    st.session_state.current_conv_title = title
    st.session_state.conv_store.save(
        st.session_state.current_conv_id,
        st.session_state.messages,
        st.session_state.memory.as_turns(),
        title,
    )

# 整体导出按钮（放最后，确保 messages 已更新）
if st.session_state.messages:
    export_text = _export_conversation(
        st.session_state.messages,
        st.session_state.get("current_conv_title", "对话记录"),
    )
    filename = f"对话_{st.session_state.current_conv_id or '未保存'}.md"
    st.download_button(
        "📥 导出全部笔记",
        data=export_text,
        file_name=filename,
        mime="text/markdown",
    )
