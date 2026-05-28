"""Central configuration, loaded from environment with defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Config:
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    embed_model: str = field(default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", "data/chroma")))
    bm25_path: Path = field(default_factory=lambda: Path(_env("BM25_PATH", "data/bm25.pkl")))
    collection: str = "ask_my_docs"

    # Chunking
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Retrieval
    dense_k: int = 20
    sparse_k: int = 20
    rrf_k: int = 60          # RRF dampening constant
    rerank_top_n: int = 5

    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def require_openai(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )


config = Config()
