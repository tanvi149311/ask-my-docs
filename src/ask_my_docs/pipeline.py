"""High-level pipeline: ingestion and question answering."""
from __future__ import annotations

from pathlib import Path

from ask_my_docs.graph.generate import generate
from ask_my_docs.ingest.loader import load_and_chunk
from ask_my_docs.retrieval import dense, hybrid, sparse
from ask_my_docs.schema import Answer


def ingest(docs_dir: Path) -> int:
    chunks = load_and_chunk(docs_dir)
    if not chunks:
        raise RuntimeError(f"No supported documents found in {docs_dir}")
    dense.build(chunks)
    sparse.build(chunks)
    return len(chunks)


def ask(question: str) -> Answer:
    contexts = hybrid.retrieve(question)
    return generate(question, contexts)
