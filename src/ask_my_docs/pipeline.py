"""High-level pipeline: ingestion and question answering."""
from __future__ import annotations

from pathlib import Path

from ask_my_docs.ingest.loader import VersionedIngester, load_and_chunk
from ask_my_docs.retrieval import dense, sparse
from ask_my_docs.schema import Answer


def ingest(docs_dir: Path, versioned: bool = True) -> int:
    """Index documents. Returns the number of newly added/updated chunks."""
    if versioned:
        ingester = VersionedIngester()
        stats = ingester.ingest(docs_dir)
        return stats["new_chunks"]

    chunks = load_and_chunk(docs_dir)
    if not chunks:
        raise RuntimeError(f"No supported documents found in {docs_dir}")
    dense.build(chunks)
    sparse.build(chunks)
    return len(chunks)


def ask(question: str, user_groups: list[str] | None = None) -> Answer:
    """Answer a question using the LangGraph RAG pipeline."""
    from ask_my_docs.graph.generate import generate
    from ask_my_docs.graph.rag_graph import get_graph
    from ask_my_docs.retrieval.hybrid import retrieve
    from ask_my_docs.security.injection import sanitize_query

    question = sanitize_query(question)

    try:
        graph = get_graph()
        final_state = graph.invoke(
            {
                "question": question,
                "user_groups": user_groups or [],
                "attempt": 0,
            }
        )

        answer = final_state.get("answer")
        if answer is not None:
            return answer

        # validate_node set validation_passed but didn't create answer — shouldn't happen
        return Answer(
            text=final_state.get("draft", _REFUSAL),
            citations=final_state.get("citations") or [],
            contexts=final_state.get("contexts") or [],
        )

    except Exception as exc:
        print(f"[pipeline] LangGraph error, falling back to direct pipeline: {exc}")
        contexts = retrieve(question)
        return generate(question, contexts)


_REFUSAL = "I could not find this in the provided documents."
