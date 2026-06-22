from langchain_core.documents import Document

from rag_agent.documents import load_builtin_documents
from rag_agent.embeddings import HashEmbedding
from rag_agent.retriever import InMemoryKnowledgeBase
from rag_agent.splitting import split_documents


def test_offline_retrieval_prefers_keyword_overlap_over_hash_similarity():
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(
        [
            Document(
                page_content="Python variables bind names to objects.",
                metadata={"source": "intro.md"},
            ),
            Document(
                page_content="list is mutable, while tuple is immutable.",
                metadata={"source": "sequence.md"},
            ),
        ]
    )

    results = kb.search("list tuple difference", k=1)

    assert results[0].document.metadata["source"] == "sequence.md"


def test_builtin_python_knowledge_retrieves_sequence_chunk_for_list_tuple_question():
    chunks = split_documents(load_builtin_documents(), chunk_size=280, chunk_overlap=40)
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(chunks)

    results = kb.search("list tuple difference", k=1)

    assert results
    assert "list" in results[0].document.page_content.lower()
    assert "tuple" in results[0].document.page_content.lower()


def test_builtin_python_knowledge_rejects_unrelated_hardware_question():
    chunks = split_documents(load_builtin_documents(), chunk_size=280, chunk_overlap=40)
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(chunks)

    results = kb.search("repair computer hardware", k=1)

    assert results == []


def test_builtin_python_knowledge_rejects_unrelated_chinese_hardware_question():
    chunks = split_documents(load_builtin_documents(), chunk_size=280, chunk_overlap=40)
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(chunks)

    results = kb.search("怎么维修电脑硬件？", k=1)

    assert results == []


def test_builtin_python_knowledge_handles_misspelled_mixed_language_function_query():
    chunks = split_documents(load_builtin_documents(), chunk_size=280, chunk_overlap=40)
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(chunks)

    results = kb.search("Funtion 怎么定义？", k=1)

    assert results
    content = results[0].document.page_content.lower()
    assert "def" in content
    assert "function" in content


def test_uploaded_domain_document_handles_misspelled_query_terms():
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "A database transaction is a logical unit of work. "
                    "Transaction isolation prevents concurrent operations from "
                    "interfering with each other."
                ),
                metadata={"source": "database.md"},
            ),
            Document(
                page_content="A database index improves lookup speed for selected columns.",
                metadata={"source": "index.md"},
            ),
        ]
    )

    results = kb.search("trasaction isolation", k=1)

    assert results
    assert results[0].document.metadata["source"] == "database.md"


def test_uploaded_document_corrects_multiple_misspelled_terms_from_vocabulary():
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "Gradient descent is an optimization algorithm that updates "
                    "model parameters in the direction of the negative gradient."
                ),
                metadata={"source": "ml.md"},
            ),
            Document(
                page_content="A confusion matrix summarizes classification results.",
                metadata={"source": "metrics.md"},
            ),
        ]
    )

    results = kb.search("gradiant desent", k=1)

    assert results
    assert results[0].document.metadata["source"] == "ml.md"


def test_knowledge_base_expands_query_using_uploaded_document_vocabulary():
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(
        [
            Document(
                page_content="Gradient descent updates parameters with a gradient.",
                metadata={"source": "ml.md"},
            )
        ]
    )

    expanded = kb.expand_query("gradiant desent")

    assert "gradient" in expanded
    assert "descent" in expanded


def test_uploaded_chinese_document_uses_keyword_retrieval():
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(
        [
            Document(
                page_content="数据库事务隔离用于避免并发操作互相干扰。",
                metadata={"source": "db_cn.md"},
            ),
            Document(
                page_content="索引可以提升数据库查询速度。",
                metadata={"source": "index_cn.md"},
            ),
        ]
    )

    results = kb.search("事务隔离有什么作用", k=1)

    assert results
    assert results[0].document.metadata["source"] == "db_cn.md"
