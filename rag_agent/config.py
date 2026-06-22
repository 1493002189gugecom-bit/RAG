from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    chat_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    openai_base_url: str
    use_fake_llm: bool

    @property
    def api_ready(self) -> bool:
        return bool(
            self.openai_api_key
            and self.openai_base_url
            and self.chat_model
            and not self.use_fake_llm
        )

    @property
    def embedding_api_ready(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_base_url and self.embedding_model)


def load_config() -> AppConfig:
    dotenv_values = _read_dotenv(Path.cwd() / ".env")
    return AppConfig(
        openai_api_key=_get_setting("OPENAI_API_KEY", dotenv_values, ""),
        chat_model=_get_setting("CHAT_MODEL", dotenv_values, "gpt-4o-mini"),
        embedding_api_key=_get_setting(
            "EMBEDDING_API_KEY",
            dotenv_values,
            _get_setting("OPENAI_API_KEY", dotenv_values, ""),
        ),
        embedding_base_url=_get_setting(
            "EMBEDDING_BASE_URL",
            dotenv_values,
            _get_setting("OPENAI_BASE_URL", dotenv_values, "https://api.openai.com/v1"),
        ),
        embedding_model=_get_setting(
            "EMBEDDING_MODEL", dotenv_values, "text-embedding-3-small"
        ),
        openai_base_url=_get_setting(
            "OPENAI_BASE_URL", dotenv_values, "https://api.openai.com/v1"
        ),
        use_fake_llm=_get_setting("USE_FAKE_LLM", dotenv_values, "false").lower()
        == "true",
    )


def _get_setting(key: str, dotenv_values: dict[str, str], default: str) -> str:
    return os.getenv(key) or dotenv_values.get(key, default)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.lstrip("\ufeff").split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
