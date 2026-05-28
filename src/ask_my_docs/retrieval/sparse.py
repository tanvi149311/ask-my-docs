"""Sparse lexical retrieval using BM25 over the same chunk corpus."""
from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from ask_my_docs.config import config
from ask_my_docs.schema import Chunk, Retrieved

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def build(chunks: list[Chunk]) -> None:
    """Build BM25 index from scratch."""
    corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    payload = {"bm25": bm25, "chunks": chunks}
    config.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.bm25_path, "wb") as f:
        pickle.dump(payload, f)


def update(to_delete: list[str], to_add: list[Chunk]) -> None:
    """Remove stale chunk IDs, add new chunks, rebuild BM25 index."""
    if config.bm25_path.exists():
        with open(config.bm25_path, "rb") as f:
            payload = pickle.load(f)
        delete_set = set(to_delete)
        existing = [c for c in payload["chunks"] if c.chunk_id not in delete_set]
    else:
        existing = []

    all_chunks = existing + to_add
    if not all_chunks:
        return

    corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(corpus)
    config.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)


def search(query: str, k: int | None = None) -> list[Retrieved]:
    k = k or config.sparse_k
    with open(config.bm25_path, "rb") as f:
        payload = pickle.load(f)
    bm25: BM25Okapi = payload["bm25"]
    chunks: list[Chunk] = payload["chunks"]

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:k]
    return [Retrieved(chunk=c, score=float(s)) for c, s in ranked]
