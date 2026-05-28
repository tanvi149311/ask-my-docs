"""LangGraph state definition for the RAG pipeline."""
from __future__ import annotations

from typing import Any


class RAGState(dict):
    """State carried between LangGraph nodes.

    Keys (all optional — nodes read with .get()):
        question          str            original user question
        user_groups       list[str]      ACL groups for the requesting user
        transformed_queries list[str]    paraphrases from query-transform node
        candidates        list           post-fusion, pre-rerank Retrieved objects
        contexts          list           top-n Retrieved objects after reranking
        draft             str            current LLM output
        citations         list[str]      chunk IDs cited in draft
        attempt           int            retry counter (starts at 0)
        answer            Answer | None  final answer once the graph terminates
        validation_passed bool           set by validate_node
    """

    # Subclassing dict (not TypedDict) avoids LangGraph serialisation edge-cases
    # with complex Python objects (Retrieved, Answer) while keeping full dict API.
    def __repr__(self) -> str:  # noqa: D401
        keys = list(self.keys())
        return f"RAGState({keys})"
