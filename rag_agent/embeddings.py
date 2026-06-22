from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from math import sqrt
from urllib import request
from urllib.error import URLError


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
    batch_size: int = 16
    max_retries: int = 2
    timeout_seconds: int = 60

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError(
                "Missing EMBEDDING_API_KEY. Configure it in environment variables or .env."
            )
        if not texts:
            return []

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

        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return [item["embedding"] for item in data["data"]]
            except URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Embedding API request failed after retries. "
                        "Try again, reduce uploaded document size, or disable API Embedding."
                    ) from exc
                time.sleep(0.8 * (attempt + 1))

        raise RuntimeError("Embedding API request failed unexpectedly.")


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
