"""Tests for versioned ingestion logic (no API key required)."""
from __future__ import annotations

from pathlib import Path

from ask_my_docs.ingest.loader import VersionedIngester, chunk_document, file_hash


def test_file_hash_is_deterministic(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    assert file_hash(f) == file_hash(f)


def test_file_hash_differs_on_change(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("version 1")
    h1 = file_hash(f)
    f.write_text("version 2")
    h2 = file_hash(f)
    assert h1 != h2


def test_chunk_document_carries_version_metadata():
    chunks = chunk_document("Hello world. This is a test sentence.", source="t.md", doc_version=3)
    assert chunks
    assert all(c.metadata["doc_version"] == 3 for c in chunks)


def test_chunk_document_carries_ingested_at():
    chunks = chunk_document("Some content here for the test.", source="t.md")
    assert all("ingested_at" in c.metadata for c in chunks)


def test_versioned_ingester_skips_unchanged_files(tmp_path, monkeypatch):
    _setup_stub_indices(tmp_path, monkeypatch)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("First document with some content.")

    manifest = tmp_path / "manifest.json"
    ingester = VersionedIngester(manifest_path=manifest)
    stats = ingester.ingest(docs_dir)
    assert stats["added"] == 1
    assert stats["skipped"] == 0

    # Second run with no changes → skip
    ingester2 = VersionedIngester(manifest_path=manifest)
    stats2 = ingester2.ingest(docs_dir)
    assert stats2["skipped"] == 1
    assert stats2["new_chunks"] == 0


def test_versioned_ingester_detects_update(tmp_path, monkeypatch):
    _setup_stub_indices(tmp_path, monkeypatch)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc = docs_dir / "a.md"
    doc.write_text("Original document content here.")

    manifest = tmp_path / "manifest.json"
    VersionedIngester(manifest_path=manifest).ingest(docs_dir)

    doc.write_text("Updated document content with entirely new information.")
    stats = VersionedIngester(manifest_path=manifest).ingest(docs_dir)
    assert stats["updated"] == 1
    assert stats["new_chunks"] > 0


def test_versioned_ingester_tombstones_deleted_file(tmp_path, monkeypatch):
    _setup_stub_indices(tmp_path, monkeypatch)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc = docs_dir / "a.md"
    doc.write_text("Document that will be deleted.")

    manifest = tmp_path / "manifest.json"
    VersionedIngester(manifest_path=manifest).ingest(docs_dir)

    doc.unlink()
    stats = VersionedIngester(manifest_path=manifest).ingest(docs_dir)
    assert stats["removed"] == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_stub_indices(tmp_path: Path, monkeypatch) -> None:
    """Stub dense/sparse update so tests don't need API keys or real index files."""
    import ask_my_docs.retrieval.dense as dense_mod
    import ask_my_docs.retrieval.sparse as sparse_mod

    monkeypatch.setattr(dense_mod, "update", lambda _d, _a: None)
    monkeypatch.setattr(sparse_mod, "update", lambda _d, _a: None)
