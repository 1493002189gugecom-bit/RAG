from rag_agent.llm import FakeLLM, OpenAICompatibleLLM


def test_fake_llm_returns_contextual_answer():
    llm = FakeLLM()

    answer = llm.generate("知识库片段：\n[片段 1]\nPython list 是可变序列。\n\n用户问题：list 是什么？")

    assert "测试模式回答" in answer
    assert "Python list 是可变序列" in answer


def test_openai_compatible_llm_requires_api_key():
    llm = OpenAICompatibleLLM(api_key="", model="demo")

    try:
        llm.generate("hello")
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("missing api key should raise RuntimeError")


def test_openai_compatible_llm_retries_transient_url_errors(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(req, timeout):
        from urllib.error import URLError

        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("transient ssl eof")
        return FakeResponse()

    monkeypatch.setattr("rag_agent.llm.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleLLM(api_key="secret", model="demo", max_retries=2)

    assert llm.generate("hello") == "ok"
    assert calls["count"] == 2
