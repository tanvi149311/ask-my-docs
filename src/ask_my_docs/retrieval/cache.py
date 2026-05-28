"""Redis-backed retrieval cache. Silently no-ops when Redis is unavailable."""
from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

from ask_my_docs.config import config
from ask_my_docs.schema import Retrieved

_client: Any = None
_client_checked = False


def _redis():
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    try:
        import redis

        r = redis.from_url(
            config.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=False,
        )
        r.ping()
        _client = r
    except Exception:
        _client = None
    return _client


def _key(query: str, filters: dict[str, Any] | None = None) -> str:
    payload = {"q": query.lower().strip(), "f": filters or {}}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:24]
    return f"ask_my_docs:cache:{digest}"


def get(query: str, filters: dict[str, Any] | None = None) -> list[Retrieved] | None:
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(_key(query, filters))
        return pickle.loads(raw) if raw else None
    except Exception:
        return None


def set(  # noqa: A001
    query: str,
    results: list[Retrieved],
    filters: dict[str, Any] | None = None,
) -> None:
    r = _redis()
    if r is None:
        return
    try:
        r.setex(_key(query, filters), config.redis_ttl, pickle.dumps(results))
    except Exception:
        pass


def invalidate(query: str, filters: dict[str, Any] | None = None) -> None:
    r = _redis()
    if r is None:
        return
    try:
        r.delete(_key(query, filters))
    except Exception:
        pass
