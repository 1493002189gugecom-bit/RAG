from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ConversationStore:
    """Persist / list / load / delete conversation histories as JSON files."""

    def __init__(self, storage_dir: str = "conversations") -> None:
        root = Path(__file__).resolve().parent.parent
        self._storage = root / storage_dir
        self._storage.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        conv_id: str,
        messages: list[dict],
        turns: list[tuple[str, str]],
        title: str,
    ) -> None:
        path = self._storage / f"{conv_id}.json"
        now = datetime.now(timezone.utc).isoformat()

        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            created_at = existing.get("created_at", now)
        else:
            created_at = now

        data = {
            "id": conv_id,
            "title": title[:30],
            "created_at": created_at,
            "updated_at": now,
            "messages": messages,
            "turns": turns,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, conv_id: str) -> dict | None:
        path = self._storage / f"{conv_id}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def list(self) -> list[dict]:
        conversations: list[dict] = []
        for path in sorted(self._storage.glob("*.json"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                conversations.append(
                    {
                        "id": data.get("id", path.stem),
                        "title": data.get("title", "未命名对话"),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return conversations

    def delete(self, conv_id: str) -> None:
        path = self._storage / f"{conv_id}.json"
        if path.exists():
            path.unlink()
