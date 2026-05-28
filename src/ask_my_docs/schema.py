"""Shared data types passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance metadata."""
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Retrieved:
    """A chunk with a relevance score attached by a retrieval/rerank stage."""
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    citations: list[str]            # chunk_ids actually cited
    contexts: list[Retrieved]       # what was sent to the LLM
