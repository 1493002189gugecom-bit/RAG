from pathlib import Path

from rag_agent.documents import load_builtin_documents, load_text_file


def test_load_text_file_reads_markdown_with_source_metadata(tmp_path):
    path = tmp_path / "python_notes.md"
    path.write_text("# Python 列表\n列表可以追加元素。", encoding="utf-8")

    docs = load_text_file(path)

    assert len(docs) == 1
    assert "列表可以追加元素" in docs[0].page_content
    assert docs[0].metadata["source"] == "python_notes.md"


def test_load_text_file_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_text("not supported", encoding="utf-8")

    try:
        load_text_file(path)
    except ValueError as exc:
        assert "仅支持" in str(exc)
    else:
        raise AssertionError("unsupported suffix should raise ValueError")


def test_load_builtin_documents_points_to_python_syntax_file():
    docs = load_builtin_documents()

    assert docs
    assert docs[0].metadata["source"] == "python_syntax.md"
    assert "Python" in docs[0].page_content
