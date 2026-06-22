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

    results = kb.search("\u600e\u4e48\u7ef4\u4fee\u7535\u8111\u786c\u4ef6\uff1f", k=1)

    assert results == []


def test_builtin_python_knowledge_handles_misspelled_mixed_language_function_query():
    chunks = split_documents(load_builtin_documents(), chunk_size=280, chunk_overlap=40)
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(chunks)

    results = kb.search("Funtion \u600e\u4e48\u5b9a\u4e49\uff1f", k=1)

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
                page_content="\u6570\u636e\u5e93\u4e8b\u52a1\u9694\u79bb\u7528\u4e8e\u907f\u514d\u5e76\u53d1\u64cd\u4f5c\u4e92\u76f8\u5e72\u6270\u3002",
                metadata={"source": "db_cn.md"},
            ),
            Document(
                page_content="\u7d22\u5f15\u53ef\u4ee5\u63d0\u5347\u6570\u636e\u5e93\u67e5\u8be2\u901f\u5ea6\u3002",
                metadata={"source": "index_cn.md"},
            ),
        ]
    )

    results = kb.search("\u4e8b\u52a1\u9694\u79bb\u6709\u4ec0\u4e48\u4f5c\u7528\uff1f", k=1)

    assert results
    assert results[0].document.metadata["source"] == "db_cn.md"


def test_uploaded_network_document_retrieves_three_way_handshake_locally():
    kb = InMemoryKnowledgeBase(HashEmbedding())
    kb.add_documents(
        [
            Document(
                page_content=(
                    "TCP \u4e09\u6b21\u63e1\u624b\u7528\u4e8e\u5efa\u7acb\u53ef\u9760\u8fde\u63a5\u3002"
                    "\u7b2c\u4e00\u6b21\u5ba2\u6237\u7aef\u53d1\u9001 SYN\uff0c"
                    "\u7b2c\u4e8c\u6b21\u670d\u52a1\u5668\u56de\u590d SYN-ACK\uff0c"
                    "\u7b2c\u4e09\u6b21\u5ba2\u6237\u7aef\u53d1\u9001 ACK\u3002"
                ),
                metadata={"source": "network_cn.md"},
            ),
            Document(
                page_content="\u8def\u7531\u5668\u6839\u636e IP \u5730\u5740\u8f6c\u53d1\u6570\u636e\u5305\u3002",
                metadata={"source": "router_cn.md"},
            ),
        ]
    )

    results = kb.search("\u4e09\u6b21\u63e1\u624b", k=1)

    assert results
    assert results[0].document.metadata["source"] == "network_cn.md"
