"""Structured JSON logging for requests, feedback, and operational events."""
from __future__ import annotations

import json
import logging
import time

_log = logging.getLogger("ask_my_docs")

if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


def _emit(level: int, **fields) -> None:
    fields.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _log.log(level, json.dumps(fields, default=str))


def log_request(
    question: str,
    answer_len: int,
    citations: int,
    latency_ms: float,
    tokens: int | None = None,
) -> None:
    _emit(
        logging.INFO,
        event="request",
        question_chars=len(question),
        answer_chars=answer_len,
        citations=citations,
        latency_ms=round(latency_ms, 1),
        tokens=tokens,
    )


def log_feedback(question: str, rating: int, comment: str = "") -> None:
    _emit(
        logging.INFO,
        event="feedback",
        question_chars=len(question),
        rating=rating,
        has_comment=bool(comment),
    )


def log_retrieval_miss(question: str) -> None:
    _emit(logging.WARNING, event="retrieval_miss", question_chars=len(question))


def log_rerank_fallback(reason: str) -> None:
    _emit(logging.WARNING, event="rerank_fallback", reason=str(reason)[:200])


def log_ingest(stats: dict) -> None:
    _emit(logging.INFO, event="ingest", **stats)
