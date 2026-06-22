from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".txt", ".md"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "python_syntax.md"


def load_text_file(path: str | Path) -> list[Document]:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 .txt 和 .md 文档。")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": file_path.name},
        )
    ]


def load_builtin_documents() -> list[Document]:
    return load_text_file(BUILTIN_KNOWLEDGE_PATH)
