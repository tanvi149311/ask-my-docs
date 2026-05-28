"""Hybrid retrieval: fuse dense + sparse with RRF, then cross-encoder rerank."""
from __future__ import annotations

from functools import lru_cache

from ask_my_docs.config import config
from ask_my_docs.schema import Retrieved


def reciprocal_rank_fusion(
    dense_hits: list[Retrieved],
    sparse_hits: list[Retrieved],
    k: int | None = None,
) -> list[Retrieved]:
    """Combine two ranked lists by Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (rrf_k + rank(d))
    """
    rrf_k = config.rrf_k
    fused: dict[str, float] = {}
    by_id: dict[str, Retrieved] = {}

    for hits in (dense_hits, sparse_hits):
        for rank, r in enumerate(hits):
            cid = r.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            by_id.setdefault(cid, r)

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ordered:
        r = by_id[cid]
        results.append(Retrieved(chunk=r.chunk, score=score))
    if k:
        results = results[:k]
    return results


@lru_cache(maxsize=1)
def _reranker():
    """Lazily load the SBERT cross-encoder (heavy import)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.reranker_model)


def rerank(query: str, candidates: list[Retrieved], top_n: int | None = None) -> list[Retrieved]:
    top_n = top_n or config.rerank_top_n
    if not candidates:
        return []
    try:
        model = _reranker()
        pairs = [(query, r.chunk.text) for r in candidates]
        scores = model.predict(pairs)
        reranked = sorted(
            (Retrieved(chunk=c.chunk, score=float(s)) for c, s in zip(candidates, scores)),
            key=lambda r: r.score,
            reverse=True,
        )
        return reranked[:top_n]
    except Exception as exc:  # noqa: BLE001 — graceful fallback per the plan
        # Fallback: keep fusion order if the reranker is unavailable.
        print(f"[rerank] falling back to fusion order: {exc}")
        return candidates[:top_n]


def retrieve(query: str) -> list[Retrieved]:
    """Full hybrid retrieval + rerank for a single query."""
    from ask_my_docs.retrieval import dense, sparse  # lazy: avoids heavy imports at module load

    dense_hits = dense.search(query)
    sparse_hits = sparse.search(query)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
    return rerank(query, fused)
