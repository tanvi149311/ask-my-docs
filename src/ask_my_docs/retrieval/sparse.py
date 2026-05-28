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
    corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    payload = {
        "bm25": bm25,
        "chunks": chunks,  # store chunks so search can reconstruct results
    }
    config.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.bm25_path, "wb") as f:
        pickle.dump(payload, f)


def search(query: str, k: int | None = None) -> list[Retrieved]:
    k = k or config.sparse_k
    with open(config.bm25_path, "rb") as f:
        payload = pickle.load(f)
    bm25: BM25Okapi = payload["bm25"]
    chunks: list[Chunk] = payload["chunks"]

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:k]
    return [Retrieved(chunk=c, score=float(s)) for c, s in ranked]
