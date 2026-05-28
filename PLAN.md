# Build Roadmap

## ✅ Iteration 0 — Walking skeleton (this commit)
End-to-end thin slice:
- Ingestion: load `.txt/.md/.pdf`, recursive chunking, deterministic chunk ids.
- Dense retrieval: OpenAI embeddings + persistent Chroma collection.
- Sparse retrieval: BM25 over the same corpus.
- Hybrid fusion: Reciprocal Rank Fusion (RRF).
- Reranking: SBERT cross-encoder, with graceful fallback to fusion order.
- Generation: OpenAI, citation-enforced prompt + post-hoc grounding check
  (answers without valid citations are withheld).
- CLI: `ingest` and `ask`.
- Tests: RRF fusion + citation extraction (pure logic, no API key needed).

## ⏳ Deferred (next iterations)
1. **LangGraph orchestration** — replace the function pipeline in
   `pipeline.py` with a `StateGraph`: transform → retrieve → rerank →
   generate → validate → (retry | refuse) edges.
2. **Query transformation** — rewrite, multi-query, metadata filters.
3. **Versioning & re-indexing** — content-hash per doc, tombstone stale chunks.
4. **Caching** — Redis query→candidates and embedding cache.
5. **Security** — prompt-injection hardening, PII redaction, per-user ACL
   filtering at retrieval time.
6. **Evaluation (RAGAS)** — golden set + faithfulness / answer_relevancy /
   context_precision / context_recall.
7. **CI gate** — GitHub Actions running eval, failing the build on regression.
8. **Serving** — FastAPI `/ask`, Dockerfile, health checks, rate limiting.
9. **Monitoring** — logging, cost/latency tracking, feedback → eval set.

## Notes / things to verify
- Model identifiers (`gpt-4o-mini`, `text-embedding-3-small`) and package
  versions are best-guess as of the build date — verify against current docs.
- The heavy ML stack (torch via sentence-transformers) was not installed in
  the build sandbox due to disk limits; reranking is covered by code +
  fallback but was not executed end-to-end here. Run it locally to confirm.
