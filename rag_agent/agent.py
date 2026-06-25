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
            return QAResponse(
                answer="知识库尚未构建，请先导入知识点文档。",
                sources=[],
                contexts=[],
            )

        context_subjects = _context_subjects_for_question(question, self.memory)
        resolved_question = _resolve_contextual_question(question, context_subjects)
        retrieval_query = _build_retrieval_query(question, self.memory, context_subjects)
        search_k = max(self.top_k * 3, 12) if context_subjects else self.top_k
        results = self.knowledge_base.search(retrieval_query, k=search_k)
        results = _rerank_contextual_results(context_subjects, results)
        results = results[: self.top_k]

        if not results:
            return QAResponse(
                answer="知识库中未找到与问题相关的内容，请补充资料后再提问。",
                sources=[],
                contexts=[],
            )

        contexts = [result.document.page_content for result in results]
        sources = _unique_sources(results)
        prompt = self._build_prompt(question, contexts, resolved_question)
        answer = self.llm.generate(prompt).strip()
        self.memory.add_turn(question, answer)
        return QAResponse(answer=answer, sources=sources, contexts=contexts)

    def _build_prompt(
        self, question: str, contexts: list[str], resolved_question: str | None = None
    ) -> str:
        context_block = "\n\n".join(
            f"[片段 {index}]\n{context}"
            for index, context in enumerate(contexts, start=1)
        )
        resolved_block = (
            f"指代消解后的问题：{resolved_question}\n"
            if resolved_question and resolved_question != question
            else ""
        )
        return (
            "你是一个专业基础课知识点问答助手。\n"
            "必须只依据“知识库片段”回答，不要引入片段外知识。\n"
            "如果用户问题中出现“它们、两者、三者、这些”等指代，"
            "请优先依据“指代消解后的问题”和历史对话理解指代对象。\n"
            "如果片段无法回答问题，请回答：知识库中未找到与问题相关的内容，请补充资料后再提问。\n\n"
            f"历史对话：\n{self.memory.format_for_prompt()}\n\n"
            f"知识库片段：\n{context_block}\n\n"
            f"用户问题：{question}\n\n"
            f"{resolved_block}"
            "请用简洁中文回答，并在需要时分点说明。"
        )


def _unique_sources(results) -> list[str]:
    sources: list[str] = []
    for result in results:
        source = result.document.metadata.get("source", "未知来源")
        if source not in sources:
            sources.append(source)
    return sources


def _build_retrieval_query(
    question: str, memory: ConversationMemory, context_subjects: list[str]
) -> str:
    if not context_subjects:
        return question
    recent_questions = memory.recent_questions(limit=len(context_subjects))
    return " ".join([*recent_questions, _resolve_contextual_question(question, context_subjects)])


def _rerank_contextual_results(context_subjects: list[str], results):
    if not results or not context_subjects:
        return results

    subject_groups = [_subject_variants(subject) for subject in context_subjects]
    scored_results = []
    for result in results:
        content = result.document.page_content.lower()
        group_hits = sum(1 for variants in subject_groups if _has_any_variant(content, variants))
        pair_hits = _subject_pair_hits(content, context_subjects)
        heading_hits = _heading_subject_hits(result.document.page_content, subject_groups)
        scored_results.append((heading_hits, pair_hits, group_hits, result))

    best_group_hits = max(group_hits for _, _, group_hits, _ in scored_results)
    if best_group_hits > 0:
        scored_results = [
            (heading_hits, pair_hits, group_hits, result)
            for heading_hits, pair_hits, group_hits, result in scored_results
            if group_hits == best_group_hits
        ]

    def rank_key(item):
        heading_hits, pair_hits, group_hits, result = item
        return (heading_hits, pair_hits, group_hits, result.score)

    return [result for _, _, _, result in sorted(scored_results, key=rank_key, reverse=True)]


def _context_subjects_for_question(question: str, memory: ConversationMemory) -> list[str]:
    if not _needs_conversation_context(question):
        return []

    limit = _subject_limit_for_question(question, len(memory.as_turns()))
    subjects: list[str] = []
    for previous_question in memory.recent_questions(limit=limit):
        subject = _extract_subject(previous_question)
        if subject and subject not in subjects:
            subjects.append(subject)
    return subjects


def _subject_limit_for_question(question: str, available: int) -> int:
    compact = _compact(question)
    if any(
        marker in compact
        for marker in [
            "三者",
            "三个",
            "前三个",
            "前面三个",
            "上面三个",
            "这三个",
            "这三者",
        ]
    ):
        return min(3, available)
    if any(
        marker in compact
        for marker in [
            "两者",
            "二者",
            "两个",
            "前两个",
            "前面两个",
            "上面两个",
            "这两个",
            "这两者",
            "前者",
            "后者",
        ]
    ):
        return min(2, available)
    return available


def _resolve_contextual_question(question: str, subjects: list[str]) -> str:
    if not subjects:
        return question
    subject_text = "、".join(subjects)
    resolved = question
    for marker in [
        "两者",
        "二者",
        "三者",
        "它们",
        "他们",
        "前面两个",
        "上面两个",
        "前两个",
        "这两个",
        "这两者",
        "前面三个",
        "上面三个",
        "前三个",
        "这三个",
        "这三者",
        "这些",
    ]:
        resolved = resolved.replace(marker, subject_text)
    if resolved == question:
        resolved = f"{subject_text}{question}"
    return resolved


def _extract_subject(question: str) -> str:
    compact = _compact(question)
    for prefix in ["什么是", "何为", "请解释", "解释一下", "介绍一下"]:
        if compact.startswith(prefix):
            compact = compact[len(prefix) :]
    for suffix in ["是什么", "是啥", "什么意思", "的定义", "指什么", "有什么特点"]:
        if compact.endswith(suffix):
            compact = compact[: -len(suffix)]
    return compact


def _needs_conversation_context(question: str) -> bool:
    compact = _compact(question)
    if not compact:
        return False

    context_markers = [
        "两者",
        "二者",
        "三者",
        "它们",
        "他们",
        "这两个",
        "这两者",
        "这三个",
        "这三者",
        "这些",
        "前者",
        "后者",
        "前面",
        "上面",
        "刚才",
        "之间",
    ]
    comparison_markers = ["关系", "区别", "不同", "相同", "联系"]
    return any(marker in compact for marker in context_markers) or (
        any(marker in compact for marker in comparison_markers)
        and any(marker in compact for marker in ["它", "这", "两", "三", "二", "之间"])
    )


def _subject_variants(subject: str) -> list[str]:
    return [subject, *_TERM_ALIASES.get(subject, [])]


def _has_any_variant(content: str, variants: list[str]) -> bool:
    return any(variant.lower() in content for variant in variants)


def _heading_subject_hits(content: str, subject_groups: list[list[str]]) -> int:
    heading = _first_heading(content).lower()
    if not heading:
        return 0
    return sum(1 for variants in subject_groups if _has_any_variant(heading, variants))


def _first_heading(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _subject_pair_hits(content: str, subjects: list[str]) -> int:
    if len(subjects) < 2:
        return 0

    hits = 0
    for left in subjects:
        for right in subjects:
            if left == right:
                continue
            for joiner in ["与", "和", "及", "/", " "]:
                if f"{left}{joiner}{right}".lower() in content:
                    hits += 1
    return hits


def _compact(text: str) -> str:
    return "".join(text.strip().lower().split()).strip("，。？！?；;：:、")


_TERM_ALIASES = {
    "类": ["class"],
    "对象": ["object"],
    "列表": ["list"],
    "元组": ["tuple"],
    "字典": ["dict"],
    "函数": ["function", "def"],
    "方法": ["method"],
    "属性": ["attribute"],
    "集合": ["set"],
    "字符串": ["str"],
    "模块": ["module"],
}
