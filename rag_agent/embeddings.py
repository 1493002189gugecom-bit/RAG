from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import sqrt
from urllib import request


@dataclass
class HashEmbedding:
    """Small deterministic embedding for offline demos and tests."""

    dimensions: int = 2048

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokenize(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)


@dataclass
class OpenAICompatibleEmbedding:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，请先在环境变量或 .env 文件中配置。")

        payload = {"model": self.model, "input": texts}
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [item["embedding"] for item in data["data"]]


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-zA-Z_]+|[\u4e00-\u9fff]", lowered)
    if words:
        return words
    return list(lowered.strip())


def _normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
