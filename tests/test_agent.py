from langchain_core.documents import Document

from rag_agent.agent import KnowledgeQAAgent
from rag_agent.memory import ConversationMemory
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


class RecordingLLM:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, prompt):
        self.last_prompt = prompt
        return "根据知识库：Python list 是可变序列。"


def test_agent_answers_only_when_knowledge_context_exists():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [Document(page_content="Python list 是可变序列。", metadata={"source": "list.md"})]
    )
    llm = RecordingLLM()
    agent = KnowledgeQAAgent(kb, llm, ConversationMemory(max_turns=2))

    response = agent.answer("list 是什么？")

    assert "list" in response.answer
    assert response.sources == ["list.md"]
    assert "Python list 是可变序列" in llm.last_prompt


def test_agent_refuses_blank_question():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    agent = KnowledgeQAAgent(kb, RecordingLLM(), ConversationMemory(max_turns=2))

    response = agent.answer("  ")

    assert response.answer == "请输入有效问题。"
    assert response.sources == []


def test_agent_refuses_when_no_retrieved_context():
    kb = InMemoryKnowledgeBase(embedding_model=KeywordEmbedding())
    kb.add_documents(
        [Document(page_content="Python dict 用于保存键值对。", metadata={"source": "dict.md"})]
    )
    agent = KnowledgeQAAgent(kb, RecordingLLM(), ConversationMemory(max_turns=2))

    response = agent.answer("路由器硬件如何维修？")

    assert response.answer == "知识库中未找到与问题相关的内容，请补充资料后再提问。"
    assert response.sources == []
