# Build Roadmap

## ✅ Iteration 0 — Walking skeleton
End-to-end thin slice:
- Ingestion: load `.txt/.md/.pdf`, recursive chunking, deterministic chunk ids.
- Dense retrieval: OpenAI embeddings + persistent Chroma collection.
- Sparse retrieval: BM25 over the same corpus.
- Hybrid fusion: Reciprocal Rank Fusion (RRF).
- Reranking: SBERT cross-encoder, with graceful fallback to fusion order.
- Generation: OpenAI, citation-enforced prompt + post-hoc grounding check.
- CLI: `ingest` and `ask`.
- Tests: RRF fusion + citation extraction (pure logic, no API key needed).

## ✅ Iteration 1 — Production hardening
Built on top of the skeleton:
- **LangGraph orchestration** — `StateGraph` with nodes:
  `transform → retrieve → rerank → generate → validate → (retry | refuse)`.
  Bounded retry loop; graceful refusal after `max_retries` exhausted.
- **Query transformation** — LLM-generated paraphrases (multi-query) for
  broader recall; falls back to original query on error.
- **Versioned ingestion** — `VersionedIngester` tracks SHA-256 per file;
  only re-embeds changed docs; tombstones stale chunk IDs.
  Incremental `update()` on both Chroma and BM25.
- **Redis caching** — query→candidates cached with TTL; silently degrades
  to no-cache when Redis is unavailable.
- **Security** — prompt-injection defense (context in XML delimiters, "treat
  as data not instructions"), ACL chunk filtering by `allowed_groups` metadata,
  query sanitization.
- **FastAPI** — `/ask`, `/ingest`, `/health`, `/feedback` endpoints.
- **Structured monitoring** — JSON-line logs for requests, feedback,
  retrieval misses, rerank fallbacks, ingest stats.
- **RAGAS evaluation** — golden set (`eval/golden_set.json`) + `run_eval.py`
  with faithfulness / answer_relevancy / context_precision / context_recall.
- **CI gate** — `.github/workflows/eval.yml` runs unit tests on every PR
  and RAGAS eval (with threshold assertions) when `OPENAI_API_KEY` is set.
- **Dockerfile** — single-stage image; health check; uvicorn entrypoint.
- **New tests** — versioning, security/ACL, graph routing (all no-API-key).

## ⏳ Iteration 2 — Next priorities
1. **Rate limiting** — add per-IP/per-user throttling to the FastAPI app.
2. **Cost tracking** — capture token usage from LLM responses and log/alert.
3. **Cohere reranker** — wire `cohere_api_key` to `rerank_node` as an
   alternative to SBERT.
4. **PII redaction** — scan retrieved chunks for PII before logging/returning.
5. **Feedback → eval set pipeline** — low-rated answers auto-added to golden set.
6. **Latency / drift alerts** — monitor rerank-fallback rate, refusal rate,
   retrieval-miss rate with threshold alerting.
7. **HTML / DOCX loading** — add `UnstructuredLoader` for `.html` and `.docx`.

## Notes / things to verify
- Model identifiers (`gpt-4o-mini`, `text-embedding-3-small`) and package
  versions are best-guess — verify against current docs before relying on them.
- RAGAS API changes frequently; if `run_eval.py` breaks, check current RAGAS docs
  for metric names and the `evaluate()` interface.
- The SBERT cross-encoder requires `sentence-transformers` + PyTorch; confirm
  end-to-end on your hardware (heavy install not exercised in CI).
- Cohere reranker is wired in config but the node still uses SBERT; swap in
  Iteration 2.
