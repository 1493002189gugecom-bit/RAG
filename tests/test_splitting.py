from langchain_core.documents import Document

from rag_agent.splitting import split_documents


def test_split_documents_preserves_metadata_and_content():
    docs = [
        Document(
            page_content="Python 列表是可变序列，元组是不可变序列，字典用于保存键值对。",
            metadata={"source": "python.md"},
        )
    ]

    chunks = split_documents(docs, chunk_size=16, chunk_overlap=4)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "python.md" for chunk in chunks)
    assert "Python" in chunks[0].page_content


def test_split_documents_drops_empty_documents():
    docs = [Document(page_content="   ", metadata={"source": "empty.md"})]

    assert split_documents(docs) == []
