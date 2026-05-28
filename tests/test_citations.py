"""Test citation extraction filters to valid chunk ids only."""
from __future__ import annotations

from ask_my_docs.graph.generate import _extract_citations


def test_extracts_valid_ids_only():
    text = "The sky is blue [doc-1-abc]. Grass is green [doc-2-def, doc-9-zzz]."
    valid = {"doc-1-abc", "doc-2-def"}
    cites = _extract_citations(text, valid)
    assert cites == ["doc-1-abc", "doc-2-def"]  # doc-9-zzz dropped as invalid


def test_no_citations_returns_empty():
    assert _extract_citations("No brackets here.", {"x"}) == []
