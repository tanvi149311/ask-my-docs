"""ACL-based filtering of retrieved chunks at query time."""
from __future__ import annotations

from ask_my_docs.schema import Retrieved


def filter_by_acl(
    candidates: list[Retrieved],
    user_groups: set[str] | frozenset[str],
) -> list[Retrieved]:
    """Remove chunks whose metadata ACL does not permit the requesting user.

    Chunks without an 'allowed_groups' key are open-access.
    Pass an empty set to skip filtering entirely (anonymous / open access).
    """
    if not user_groups:
        return candidates

    out: list[Retrieved] = []
    for r in candidates:
        allowed = r.chunk.metadata.get("allowed_groups")
        if allowed is None or any(g in allowed for g in user_groups):
            out.append(r)
    return out
