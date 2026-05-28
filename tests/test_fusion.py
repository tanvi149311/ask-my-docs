"""Tests for pure logic that needs no API keys: RRF fusion and citation parsing."""
from __future__ import annotations

from ask_my_docs.retrieval.hybrid import reciprocal_rank_fusion
from ask_my_docs.schema import Chunk, Retrieved


def _r(cid: str, score: float = 1.0) -> Retrieved:
    return Retrieved(chunk=Chunk(chunk_id=cid, text=cid, source="t"), score=score)


def test_rrf_rewards_agreement():
    # A appears high in both lists -> should rank first after fusion.
    dense = [_r("A"), _r("B"), _r("C")]
    sparse = [_r("A"), _r("C"), _r("D")]
    fused = reciprocal_rank_fusion(dense, sparse)
    assert fused[0].chunk.chunk_id == "A"


def test_rrf_unions_all_ids():
    dense = [_r("A"), _r("B")]
    sparse = [_r("C"), _r("D")]
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = {r.chunk.chunk_id for r in fused}
    assert ids == {"A", "B", "C", "D"}


def test_rrf_respects_k():
    dense = [_r("A"), _r("B"), _r("C")]
    sparse = [_r("D"), _r("E"), _r("F")]
    fused = reciprocal_rank_fusion(dense, sparse, k=2)
    assert len(fused) == 2
