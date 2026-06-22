from __future__ import annotations

from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    RecursiveCharacterTextSplitter = None


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    clean_docs = [
        Document(page_content=doc.page_content.strip(), metadata=dict(doc.metadata))
        for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]
    if not clean_docs:
        return []

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
        )
        return splitter.split_documents(clean_docs)

    return _simple_split_documents(clean_docs, chunk_size, chunk_overlap)


def _simple_split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    chunks: list[Document] = []
    step = max(1, chunk_size - chunk_overlap)
    for document in documents:
        text = document.page_content
        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(document.metadata),
                    )
                )
            if start + chunk_size >= len(text):
                break
    return chunks
