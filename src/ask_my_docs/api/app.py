"""FastAPI application: /ask, /ingest, /health, /feedback."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ask_my_docs.monitoring.logger import log_feedback, log_ingest, log_request, log_retrieval_miss

app = FastAPI(title="Ask My Docs", version="0.2.0")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    user_groups: list[str] = []


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float


class IngestRequest(BaseModel):
    docs_dir: str = "data/docs"


class IngestResponse(BaseModel):
    new_chunks: int
    details: dict[str, Any]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    comment: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    from ask_my_docs.pipeline import ask as _ask

    t0 = time.monotonic()
    answer = _ask(req.question, user_groups=req.user_groups)
    latency_ms = (time.monotonic() - t0) * 1000

    if not answer.citations:
        log_retrieval_miss(req.question)

    log_request(
        question=req.question,
        answer_len=len(answer.text),
        citations=len(answer.citations),
        latency_ms=latency_ms,
    )

    return AskResponse(
        answer=answer.text,
        citations=answer.citations,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    docs_path = Path(req.docs_dir)
    if not docs_path.exists():
        raise HTTPException(status_code=404, detail=f"docs_dir not found: {req.docs_dir}")

    from ask_my_docs.ingest.loader import VersionedIngester

    try:
        stats = VersionedIngester().ingest(docs_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    log_ingest(stats)
    return IngestResponse(new_chunks=stats.get("new_chunks", 0), details=stats)


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")
    log_feedback(question=req.question, rating=req.rating, comment=req.comment)
    return {"status": "recorded"}
