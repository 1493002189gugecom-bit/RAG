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
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing api key should raise RuntimeError")
