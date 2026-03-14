"""Filesystem helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_file(path: str | Path) -> str:
    hasher = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_doc_id(path: str | Path) -> str:
    normalized = str(Path(path).resolve()).lower().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()[:16]
