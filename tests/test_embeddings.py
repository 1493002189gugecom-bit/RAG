from rag_agent.embeddings import HashEmbedding, OpenAICompatibleEmbedding


def test_hash_embedding_is_deterministic_and_sized():
    embedding = HashEmbedding(dimensions=16)

    first = embedding.embed_query("Python list")
    second = embedding.embed_query("Python list")

    assert first == second
    assert len(first) == 16
    assert any(value != 0 for value in first)


def test_hash_embedding_embeds_documents():
    embedding = HashEmbedding(dimensions=8)

    vectors = embedding.embed_documents(["Python list", "Python dict"])

    assert len(vectors) == 2
    assert all(len(vector) == 8 for vector in vectors)


def test_openai_compatible_embedding_requires_api_key():
    embedding = OpenAICompatibleEmbedding(api_key="", model="demo")

    try:
        embedding.embed_query("Python")
    except RuntimeError as exc:
        assert "EMBEDDING_API_KEY" in str(exc)
    else:
        raise AssertionError("missing api key should raise RuntimeError")


def test_openai_compatible_embedding_batches_document_requests(monkeypatch):
    request_sizes = []

    class FakeResponse:
        def __init__(self, count):
            self.count = count

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            items = [
                {"embedding": [float(index), 1.0]}
                for index in range(self.count)
            ]
            import json

            return json.dumps({"data": items}).encode("utf-8")

    def fake_urlopen(req, timeout):
        import json

        payload = json.loads(req.data.decode("utf-8"))
        request_sizes.append(len(payload["input"]))
        return FakeResponse(len(payload["input"]))

    monkeypatch.setattr("rag_agent.embeddings.request.urlopen", fake_urlopen)
    embedding = OpenAICompatibleEmbedding(
        api_key="secret",
        model="embedding-3",
        base_url="https://example.test/v1",
        batch_size=2,
    )

    vectors = embedding.embed_documents(["a", "b", "c", "d", "e"])

    assert request_sizes == [2, 2, 1]
    assert len(vectors) == 5


def test_openai_compatible_embedding_retries_transient_url_errors(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return b'{"data":[{"embedding":[0.1,0.2]}]}'

    def fake_urlopen(req, timeout):
        from urllib.error import URLError

        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("transient ssl eof")
        return FakeResponse()

    monkeypatch.setattr("rag_agent.embeddings.request.urlopen", fake_urlopen)
    embedding = OpenAICompatibleEmbedding(
        api_key="secret",
        model="embedding-3",
        base_url="https://example.test/v1",
        max_retries=2,
    )

    assert embedding.embed_query("hello") == [0.1, 0.2]
    assert calls["count"] == 2


def test_local_sentence_transformer_embedding_uses_injected_model():
    from rag_agent.embeddings import LocalSentenceTransformerEmbedding

    class FakeModel:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            return [[float(len(text)), 1.0] for text in texts]

    embedding = LocalSentenceTransformerEmbedding(model=FakeModel())

    assert embedding.embed_query("abc") == [3.0, 1.0]
    assert embedding.embed_documents(["a", "abcd"]) == [[1.0, 1.0], [4.0, 1.0]]


def test_local_sentence_transformer_embedding_reports_missing_dependency(monkeypatch):
    from rag_agent.embeddings import LocalSentenceTransformerEmbedding

    def fake_import(name):
        raise ImportError("missing")

    monkeypatch.setattr("rag_agent.embeddings.import_module", fake_import)
    embedding = LocalSentenceTransformerEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    try:
        embedding.embed_query("hello")
    except RuntimeError as exc:
        assert "sentence-transformers" in str(exc)
    else:
        raise AssertionError("missing dependency should raise RuntimeError")
