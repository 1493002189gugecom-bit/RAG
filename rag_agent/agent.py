from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .memory import ConversationMemory
from .retriever import InMemoryKnowledgeBase


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class QAResponse:
    answer: str
    sources: list[str]
    contexts: list[str]


class KnowledgeQAAgent:
    def __init__(
        self,
        knowledge_base: InMemoryKnowledgeBase,
        llm: LLMClient,
        memory: ConversationMemory,
        top_k: int = 4,
    ):
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.memory = memory
        self.top_k = top_k

    def answer(self, question: str) -> QAResponse:
        question = question.strip()
        if not question:
            return QAResponse(answer="请输入有效问题。", sources=[], contexts=[])
        if self.knowledge_base.is_empty:
            return QAResponse(answer="知识库尚未构建，请先导入知识点文档。", sources=[], contexts=[])

        results = self.knowledge_base.search(question, k=self.top_k)
        if not results:
            return QAResponse(
                answer="知识库中未找到与问题相关的内容，请补充资料后再提问。",
                sources=[],
                contexts=[],
            )

        contexts = [result.document.page_content for result in results]
        sources = _unique_sources(results)
        prompt = self._build_prompt(question, contexts)
        answer = self.llm.generate(prompt).strip()
        self.memory.add_turn(question, answer)
        return QAResponse(answer=answer, sources=sources, contexts=contexts)

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(
            f"[片段 {index}]\n{context}" for index, context in enumerate(contexts, start=1)
        )
        return (
            "你是一个专业基础课知识点问答助手。\n"
            "必须只依据“知识库片段”回答，不要引入片段外知识。\n"
            "如果片段无法回答问题，请回答：知识库中未找到与问题相关的内容，请补充资料后再提问。\n\n"
            f"历史对话：\n{self.memory.format_for_prompt()}\n\n"
            f"知识库片段：\n{context_block}\n\n"
            f"用户问题：{question}\n\n"
            "请用简洁中文回答，并在需要时分点说明。"
        )


def _unique_sources(results) -> list[str]:
    sources: list[str] = []
    for result in results:
        source = result.document.metadata.get("source", "未知来源")
        if source not in sources:
            sources.append(source)
    return sources
