"""Text helpers for extraction, chunking, and retrieval."""

from __future__ import annotations

import math
import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"[ \t]+")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\+\./]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s\-\+]", " ", lowered)
    return normalize_whitespace(lowered)


def estimate_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def sentence_split(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text.strip()]


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def truncate(text: str, limit: int = 240) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def page_list(start_page: int, end_page: int) -> list[int]:
    return list(range(start_page, end_page + 1))


def safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return math.fsum(values_list) / len(values_list)
