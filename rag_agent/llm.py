from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib import request


@dataclass
class FakeLLM:
    """Deterministic LLM for tests and offline classroom demos."""

    def generate(self, prompt: str) -> str:
        marker = "[片段 1]"
        if marker in prompt:
            context = prompt.split(marker, 1)[1].strip().split("\n\n[片段", 1)[0].strip()
            header_index = context.find("\n## ")
            if header_index != -1:
                context = context[header_index:].strip()
            return f"测试模式回答：{context}"
        return "测试模式回答：知识库中未找到可用片段。"


@dataclass
class OpenAICompatibleLLM:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    max_retries: int = 2

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，请先在环境变量或 .env 文件中配置。")

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except URLError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(0.8 * (attempt + 1))
        return data["choices"][0]["message"]["content"]
