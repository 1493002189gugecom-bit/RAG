from rag_agent.config import load_config


def test_config_marks_api_ready_when_key_url_and_model_exist(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=secret",
                "OPENAI_BASE_URL=https://example.test/v1",
                "CHAT_MODEL=demo-chat",
                "EMBEDDING_API_KEY=embedding-secret",
                "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
                "EMBEDDING_MODEL=demo-embedding",
                "USE_FAKE_LLM=false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("USE_FAKE_LLM", raising=False)

    config = load_config()

    assert config.api_ready is True
    assert config.embedding_api_ready is True
    assert config.embedding_api_key == "embedding-secret"
    assert config.embedding_base_url == "https://embedding.example.test/v1"


def test_config_can_force_offline_mode_even_with_key(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=secret",
                "OPENAI_BASE_URL=https://example.test/v1",
                "CHAT_MODEL=demo-chat",
                "USE_FAKE_LLM=true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("USE_FAKE_LLM", raising=False)

    config = load_config()

    assert config.api_ready is False
