# Ask My Docs — Production RAG (Walking Skeleton)

A domain-specific "Ask My Docs" RAG system. This is the **minimal walking
skeleton**: a thin end-to-end slice that ingests documents and answers
questions with hybrid retrieval, cross-encoder reranking, and enforced
citations. Later iterations add evaluation (RAGAS), a CI gate, query
transformation, caching, security, and monitoring.

## Pipeline (this skeleton)

```
docs ─▶ ingest (chunk + embed) ─▶ Chroma (dense) + BM25 (sparse)
                                        │
query ─▶ hybrid retrieve (RRF) ─▶ cross-encoder rerank ─▶ LLM + citations
```

## Stack

- **Orchestration:** LangChain / LangGraph (added in later iterations)
- **Vector store:** ChromaDB
- **Sparse retrieval:** rank-bm25
- **Reranking:** SBERT cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Generation:** OpenAI

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env        # add your OPENAI_API_KEY
python -m ask_my_docs.cli ingest data/docs
python -m ask_my_docs.cli ask "your question here"
```

## Status

Walking skeleton — see `PLAN.md` for the full 12-step roadmap and what is
intentionally deferred.
