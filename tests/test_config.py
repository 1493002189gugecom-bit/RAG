from rag_agent.config import load_config


def test_load_config_reads_dotenv_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=from-file\nCHAT_MODEL=demo-chat\nUSE_FAKE_LLM=true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("USE_FAKE_LLM", raising=False)

    config = load_config()

    assert config.openai_api_key == "from-file"
    assert config.chat_model == "demo-chat"
    assert config.use_fake_llm is True


def test_load_config_handles_utf8_bom_dotenv_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("\ufeffOPENAI_API_KEY=from-bom-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = load_config()

    assert config.openai_api_key == "from-bom-file"


def test_load_config_reads_local_embedding_settings(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EMBEDDING_BACKEND=local",
                "LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5",
                "LOCAL_EMBEDDING_DEVICE=cuda",
                "PRELOAD_LOCAL_EMBEDDING=true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("LOCAL_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("PRELOAD_LOCAL_EMBEDDING", raising=False)

    config = load_config()

    assert config.embedding_backend == "local"
    assert config.local_embedding_model == "BAAI/bge-small-zh-v1.5"
    assert config.local_embedding_device == "cuda"
    assert config.preload_local_embedding is True
