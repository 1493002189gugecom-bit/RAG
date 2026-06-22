from __future__ import annotations

from langchain_core.documents import Document

import streamlit as st

from rag_agent.agent import KnowledgeQAAgent
from rag_agent.config import load_config
from rag_agent.documents import load_builtin_documents
from rag_agent.embeddings import HashEmbedding, OpenAICompatibleEmbedding
from rag_agent.llm import FakeLLM, OpenAICompatibleLLM
from rag_agent.memory import ConversationMemory
from rag_agent.retriever import InMemoryKnowledgeBase
from rag_agent.splitting import split_documents


st.set_page_config(page_title="知识点问答小知识库 Agent", page_icon="📚")


def build_documents_from_uploads(uploaded_files) -> list[Document]:
    documents: list[Document] = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name},
                )
            )
    return documents


def create_knowledge_base(documents: list[Document], use_api_embedding: bool):
    config = load_config()
    embedding = (
        OpenAICompatibleEmbedding(
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
        )
        if use_api_embedding
        else HashEmbedding()
    )
    chunks = split_documents(documents, chunk_size=280, chunk_overlap=40)
    knowledge_base = InMemoryKnowledgeBase(embedding_model=embedding)
    knowledge_base.add_documents(chunks)
    return knowledge_base, chunks


def create_llm():
    config = load_config()
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
    if "knowledge_base" not in st.session_state:
        docs = load_builtin_documents()
        kb, chunks = create_knowledge_base(docs, use_api_embedding=False)
        st.session_state.knowledge_base = kb
        st.session_state.chunk_count = len(chunks)


initialize_state()
config = load_config()

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
    use_api_embedding = st.checkbox(
        "使用 API Embedding",
        value=False,
        disabled=not config.embedding_api_ready,
        help=(
            "需要配置 EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL。"
            "DeepSeek Chat Key 不一定提供 Embedding endpoint。"
        ),
    )
    if not config.embedding_api_ready:
        st.caption(
            "尚未配置 API Embedding。系统仍会使用本地 HashEmbedding + BM25 + "
            "关键词匹配 + 编辑距离纠错进行混合检索。"
        )
    uploaded_files = st.file_uploader(
        "上传知识库文档（.txt / .md）",
        type=["txt", "md"],
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
                kb, chunks = create_knowledge_base(documents, use_api_embedding)
                st.session_state.knowledge_base = kb
                st.session_state.chunk_count = len(chunks)
                st.session_state.memory = ConversationMemory(max_turns=4)
                st.session_state.messages = []
                mode = "API Embedding" if use_api_embedding else "本地混合检索"
                st.success(f"知识库已重建：共 {len(chunks)} 个知识块，检索模式：{mode}。")
        except Exception as exc:
            st.error(f"未能建立知识库：{exc}")

    if st.button("清空对话", use_container_width=True):
        st.session_state.memory = ConversationMemory(max_turns=4)
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("来源：" + "，".join(message["sources"]))

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
            response = agent.answer(question)
        st.markdown(response.answer)
        if response.sources:
            st.caption("来源：" + "，".join(response.sources))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        }
    )
