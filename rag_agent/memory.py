from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """Keep a fixed-size window of recent user/assistant turns."""

    max_turns: int = 3
    _turns: list[tuple[str, str]] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return
        self._turns.append((question, answer))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def as_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for question, answer in self._turns:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

    def as_turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def recent_questions(self, limit: int = 2) -> list[str]:
        return [question for question, _ in self._turns[-limit:]]

    def format_for_prompt(self) -> str:
        if not self._turns:
            return "无"
        lines: list[str] = []
        for index, (question, answer) in enumerate(self._turns, start=1):
            lines.append(f"第 {index} 轮用户：{question}")
            lines.append(f"第 {index} 轮助手：{answer}")
        return "\n".join(lines)
