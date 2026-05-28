"""Prompt-injection defense: delimit retrieved context so the LLM treats it as data."""
from __future__ import annotations

_OPEN = "<retrieved_context>\n"
_CLOSE = "\n</retrieved_context>"

_INJECTION_HINTS = [
    "ignore previous instructions",
    "disregard the above",
    "forget your instructions",
    "new instructions:",
    "you are now",
    "act as ",
    "\nsystem:",
    "\nassistant:",
    "override your",
]

_MAX_QUERY_LEN = 2000


def wrap_context(text: str) -> str:
    """Wrap context in XML delimiters so the LLM treats it as reference data, not commands."""
    return f"{_OPEN}{text}{_CLOSE}"


def has_injection_hint(text: str) -> bool:
    """Return True if the text contains obvious prompt-injection patterns."""
    lower = text.lower()
    return any(hint in lower for hint in _INJECTION_HINTS)


def sanitize_query(query: str) -> str:
    """Strip whitespace and truncate to a safe length."""
    return query.strip()[:_MAX_QUERY_LEN]
