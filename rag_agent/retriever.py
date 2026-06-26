from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log, sqrt
from typing import Protocol

from langchain_core.documents import Document


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


class InMemoryKnowledgeBase:
    """Hybrid in-memory retriever: vector + BM25 + keyword + edit distance."""

    def __init__(self, embedding_model: EmbeddingModel, min_score: float = 0.15):
        self.embedding_model = embedding_model
        self.min_score = min_score
        self._documents: list[Document] = []
        self._vectors: list[list[float]] = []
        self._doc_tokens: list[list[str]] = []
        self._document_frequency: dict[str, int] = {}
        self._vocabulary: set[str] = set()
        self._average_doc_length = 0.0

    @property
    def is_empty(self) -> bool:
        return not self._documents

    def add_documents(self, documents: list[Document]) -> None:
        clean_docs = [
            Document(page_content=doc.page_content.strip(), metadata=dict(doc.metadata))
            for doc in documents
            if doc.page_content and doc.page_content.strip()
        ]
        if not clean_docs:
            return

        vectors = self.embedding_model.embed_documents(
            [doc.page_content for doc in clean_docs]
        )
        self._documents.extend(clean_docs)
        self._vectors.extend(vectors)
        self._rebuild_lexical_index()

    def expand_query(self, query: str) -> str:
        focus_terms = _extract_focus_terms(query)
        tokens = _tokenize(query)
        for term in focus_terms:
            tokens.append(term)
            tokens.extend(_tokenize(term))

        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            corrected = _correct_term_from_vocabulary(token, self._vocabulary)
            if corrected != token:
                expanded.append(corrected)
        expanded.extend(_domain_synonyms(tokens))
        return " ".join(dict.fromkeys(expanded))

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        query = query.strip()
        if not query or self.is_empty:
            return []

        expanded_query = self.expand_query(query)
        query_tokens = _tokenize(expanded_query)
        query_vector = self.embedding_model.embed_query(query)

        results: list[SearchResult] = []
        for index, (document, vector) in enumerate(zip(self._documents, self._vectors)):
            semantic_score = _cosine_similarity(query_vector, vector)
            bm25_score = self._bm25_score(query_tokens, self._doc_tokens[index])
            keyword_score = _keyword_score(query_tokens, self._doc_tokens[index])
            edit_score = _edit_distance_score(query_tokens, self._doc_tokens[index])
            score = (
                0.45 * semantic_score
                + 0.35 * bm25_score
                + 0.15 * keyword_score
                + 0.05 * edit_score
            )

            has_lexical_evidence = (
                bm25_score > 0 or keyword_score > 0 or edit_score >= 0.82
            )
            has_semantic_evidence = semantic_score >= 0.55
            if score >= self.min_score and (has_lexical_evidence or has_semantic_evidence):
                results.append(SearchResult(document=document, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:k]

    def _rebuild_lexical_index(self) -> None:
        self._doc_tokens = [_tokenize(document.page_content) for document in self._documents]
        self._vocabulary = {
            token
            for tokens in self._doc_tokens
            for token in tokens
            if _is_indexable_token(token)
        }
        self._document_frequency = {}
        for tokens in self._doc_tokens:
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

        if self._doc_tokens:
            self._average_doc_length = sum(len(tokens) for tokens in self._doc_tokens) / len(
                self._doc_tokens
            )
        else:
            self._average_doc_length = 0.0

    def _bm25_score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        if not query_tokens or not document_tokens or not self._average_doc_length:
            return 0.0

        k1 = 1.5
        b = 0.75
        raw_score = 0.0
        document_length = len(document_tokens)
        document_count = len(self._documents)
        for token in set(query_tokens):
            frequency = document_tokens.count(token)
            if frequency == 0:
                continue
            doc_frequency = self._document_frequency.get(token, 0)
            idf = log(1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * document_length / self._average_doc_length
            )
            raw_score += idf * (frequency * (k1 + 1)) / denominator
        return raw_score / (raw_score + 1.0) if raw_score > 0 else 0.0


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    for match in re.findall(r"[a-zA-Z_]+|[0-9]+|[\u4e00-\u9fff]+", lowered):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            tokens.extend(_cjk_tokens(match))
        else:
            tokens.append(match)
    return [token for token in tokens if token and token not in _STOPWORDS]


def _cjk_tokens(text: str) -> list[str]:
    if len(text) <= 2:
        return [text]
    tokens = [text]
    for size in (2, 3, 4):
        tokens.extend(text[index : index + size] for index in range(0, len(text) - size + 1))
    return tokens


def _is_indexable_token(token: str) -> bool:
    if token in _STOPWORDS:
        return False
    if token in _CJK_TECH_TERMS:
        return True
    if token.isascii():
        return len(token) >= 3
    return len(token) >= 2


def _extract_focus_terms(query: str) -> list[str]:
    compact = re.sub(r"[\s，。？！?；;：:、]+", "", query.strip().lower())
    if not compact:
        return []

    patterns = [
        r"^(?:什么是|何为|解释一下|请解释|介绍一下)(?P<term>[\u4e00-\u9fffa-zA-Z0-9_]+)$",
        r"^(?P<term>[\u4e00-\u9fffa-zA-Z0-9_]+)(?:是什么|是啥|什么意思|的定义|指什么)$",
    ]
    terms: list[str] = []
    for pattern in patterns:
        match = re.match(pattern, compact)
        if match:
            term = match.group("term")
            if term and term not in _STOPWORDS:
                terms.append(term)
    return terms


def _correct_term_from_vocabulary(term: str, vocabulary: set[str]) -> str:
    if not term.isascii() or len(term) < 4 or not vocabulary:
        return term
    ascii_vocabulary = [token for token in vocabulary if token.isascii() and len(token) >= 4]
    if not ascii_vocabulary:
        return term

    best = max(
        ascii_vocabulary,
        key=lambda candidate: SequenceMatcher(None, term, candidate).ratio(),
    )
    ratio = SequenceMatcher(None, term, best).ratio()
    if ratio >= 0.78:
        return best
    return term


def _domain_synonyms(tokens: list[str]) -> list[str]:
    joined = " ".join(tokens)
    token_set = set(tokens)
    synonyms: list[str] = []
    if any(keyword in joined for keyword in ["\u5b9a\u4e49", "\u51fd\u6570", "\u65b9\u6cd5"]):
        synonyms.extend(["def", "function"])
    if "\u7c7b" in token_set:
        synonyms.extend(["class", "object"])
    if "\u5bf9\u8c61" in token_set:
        synonyms.append("object")
    if "\u5c5e\u6027" in token_set:
        synonyms.append("attribute")
    if "\u65b9\u6cd5" in token_set:
        synonyms.append("method")
    if "\u533a\u522b" in joined:
        synonyms.extend(["difference", "compare"])
    if "\u5217\u8868" in joined:
        synonyms.append("list")
    if "\u5143\u7ec4" in joined:
        synonyms.append("tuple")
    if "\u5b57\u5178" in joined:
        synonyms.append("dict")
    return synonyms


def _keyword_score(query_tokens: list[str], document_tokens: list[str]) -> float:
    useful_query_tokens = [token for token in query_tokens if _is_indexable_token(token)]
    if not useful_query_tokens or not document_tokens:
        return 0.0
    document_set = set(document_tokens)
    hits = sum(1 for token in useful_query_tokens if token in document_set)
    return hits / len(useful_query_tokens)


def _edit_distance_score(query_tokens: list[str], document_tokens: list[str]) -> float:
    useful_query_tokens = [
        token for token in query_tokens if token.isascii() and len(token) >= 4
    ]
    useful_document_tokens = [
        token for token in set(document_tokens) if token.isascii() and len(token) >= 4
    ]
    if not useful_query_tokens or not useful_document_tokens:
        return 0.0

    best_scores = []
    for query_token in useful_query_tokens:
        best_scores.append(
            max(
                SequenceMatcher(None, query_token, doc_token).ratio()
                for doc_token in useful_document_tokens
            )
        )
    return sum(best_scores) / len(best_scores)


def highlight_terms(text: str, query: str) -> str:
    """将 query 中的关键词在 text 中用 **粗体** 标记。

    用于在 UI 中展示检索命中的位置。复用 _tokenize 和 _is_indexable_token
    来确定哪些词值得标注。
    """
    tokens = _tokenize(query)
    for token in sorted(set(tokens), key=len, reverse=True):
        if _is_indexable_token(token):
            text = text.replace(token, f"**{token}**")
    return text


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "how",
    "is",
    "of",
    "the",
    "to",
    "what",
    "with",
    "\u4ec0\u4e48",
    "\u600e\u4e48",
    "\u5982\u4f55",
    "\u6709\u4ec0\u4e48",
    "\u4f5c\u7528",
}

_CJK_TECH_TERMS = {
    "\u7c7b",
}
