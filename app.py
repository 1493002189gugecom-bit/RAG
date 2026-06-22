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


st.set_page_config(page_title="Python Syntax Knowledge Agent", page_icon="PY")


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

st.title("Python Syntax Knowledge Agent")
st.caption("RAG demo for Python syntax. Answers are constrained by the local knowledge base.")

if config.api_ready:
    st.success(
        "Real API mode is enabled for chat responses. Retrieval uses hybrid "
        "vector + BM25 + keyword + edit-distance scoring."
    )
elif not config.openai_api_key:
    st.info(
        "No OPENAI_API_KEY is configured. The app is running in offline demo mode "
        "with HashEmbedding and FakeLLM, so it can test UI and retrieval without a paid API."
    )
else:
    st.warning(
        "An API key is configured, but USE_FAKE_LLM=true or a required model/base URL "
        "setting is missing. The app is staying in offline demo mode."
    )

with st.sidebar:
    st.subheader("Knowledge Base")
    st.write(f"Current chunk count: {st.session_state.get('chunk_count', 0)}")
    use_api_embedding = st.checkbox(
        "Use API Embedding",
        value=False,
        disabled=not config.embedding_api_ready,
        help=(
            "Requires EMBEDDING_API_KEY and EMBEDDING_BASE_URL. DeepSeek Chat keys "
            "do not necessarily provide an embeddings endpoint."
        ),
    )
    if not config.embedding_api_ready:
        st.caption(
            "API Embedding is not configured. The app still uses hybrid retrieval "
            "with local vectors, BM25, keyword matching, and edit-distance correction."
        )
    uploaded_files = st.file_uploader(
        "Upload .txt / .md documents",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Rebuild Knowledge Base", use_container_width=True):
        try:
            documents = (
                build_documents_from_uploads(uploaded_files)
                if uploaded_files
                else load_builtin_documents()
            )
            if not documents:
                st.warning("Document content is empty. Please upload a valid file.")
            else:
                kb, chunks = create_knowledge_base(documents, use_api_embedding)
                st.session_state.knowledge_base = kb
                st.session_state.chunk_count = len(chunks)
                st.session_state.memory = ConversationMemory(max_turns=4)
                st.session_state.messages = []
                st.success(f"Knowledge base rebuilt with {len(chunks)} chunks.")
        except Exception as exc:
            st.error(f"Failed to build knowledge base: {exc}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.memory = ConversationMemory(max_turns=4)
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

question = st.chat_input("Example: What is the difference between list and tuple?")
if question is not None:
    agent = KnowledgeQAAgent(
        st.session_state.knowledge_base,
        create_llm(),
        st.session_state.memory,
    )
    response = agent.answer(question)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        }
    )
    st.rerun()
