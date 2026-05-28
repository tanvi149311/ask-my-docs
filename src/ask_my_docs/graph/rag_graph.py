"""Compiled LangGraph StateGraph for the RAG pipeline.

Graph topology:
    transform → retrieve → rerank → generate → validate
                                                    │
                              ┌─────────────────────┼──────────────┐
                              ▼                     ▼              ▼
                           retry (→ generate)     END           refuse → END
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from ask_my_docs.graph.nodes import (
    generate_node,
    refuse_node,
    rerank_node,
    retrieve_node,
    route_after_validate,
    transform_node,
    validate_node,
)
from ask_my_docs.graph.state import RAGState


@lru_cache(maxsize=1)
def get_graph():
    """Return the compiled RAG graph (compiled once per process)."""
    builder = StateGraph(RAGState)

    builder.add_node("transform", transform_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("refuse", refuse_node)

    builder.set_entry_point("transform")
    builder.add_edge("transform", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"end": END, "retry": "generate", "refuse": "refuse"},
    )
    builder.add_edge("refuse", END)

    return builder.compile()
