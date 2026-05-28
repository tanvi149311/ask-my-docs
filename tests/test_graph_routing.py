"""Tests for LangGraph routing and node logic that needs no API calls."""
from __future__ import annotations

from ask_my_docs.graph.nodes import route_after_validate, validate_node
from ask_my_docs.schema import Chunk, Retrieved


def _r(cid: str) -> Retrieved:
    return Retrieved(chunk=Chunk(chunk_id=cid, text="text", source="t"), score=1.0)


# Routing logic

def test_route_end_when_validation_passed():
    assert route_after_validate({"validation_passed": True, "attempt": 1}) == "end"


def test_route_retry_within_max_retries():
    assert route_after_validate({"validation_passed": False, "attempt": 1}) == "retry"


def test_route_refuse_after_max_retries():
    assert route_after_validate({"validation_passed": False, "attempt": 99}) == "refuse"


def test_route_retry_at_zero_attempts():
    assert route_after_validate({"validation_passed": False, "attempt": 0}) == "retry"


# validate_node

def test_validate_passes_for_answer_with_citations():
    contexts = [_r("doc-1-abc")]
    state = {
        "draft": "The sky is blue [doc-1-abc].",
        "citations": ["doc-1-abc"],
        "contexts": contexts,
    }
    result = validate_node(state)
    assert result["validation_passed"] is True
    assert result["answer"] is not None


def test_validate_passes_for_refusal_phrase():
    state = {
        "draft": "I could not find this in the provided documents.",
        "citations": [],
        "contexts": [],
    }
    result = validate_node(state)
    assert result["validation_passed"] is True


def test_validate_fails_for_answer_without_citations():
    state = {
        "draft": "The sky is definitely blue.",
        "citations": [],
        "contexts": [_r("x")],
    }
    result = validate_node(state)
    assert result["validation_passed"] is False
    assert "answer" not in result
