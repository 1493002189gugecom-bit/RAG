from langchain_core.documents import Document

from rag_agent.retriever import InMemoryKnowledgeBase


class KeywordEmbedding:
    keywords = ["list", "tuple", "dict", "function", "python"]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lowered = text.lower()
        return [float(lowered.count(keyword)) for keyword in self.keywords]


def test_retriever_returns_semantically_relevant_chunks():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [
            Document(page_content="Python list 是可变序列。", metadata={"source": "a.md"}),
            Document(page_content="Python dict 用于保存键值对。", metadata={"source": "b.md"}),
        ]
    )

    results = kb.search("list 有什么特点？", k=1)

    assert len(results) == 1
    assert "list" in results[0].document.page_content
    assert results[0].score > 0


def test_retriever_uses_fuzzy_match_when_embedding_score_is_zero():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [Document(page_content="function 使用 def 关键字定义。", metadata={"source": "function.md"})]
    )

    results = kb.search("funtion def", k=1)

    assert len(results) == 1
    assert results[0].document.metadata["source"] == "function.md"


def test_retriever_handles_empty_store():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())

    assert kb.search("list", k=3) == []
