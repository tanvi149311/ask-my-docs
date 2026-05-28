"""Load documents from disk and split them into chunks with metadata."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ask_my_docs.config import config
from ask_my_docs.schema import Chunk

SUPPORTED = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_text(path)


def discover(docs_dir: Path) -> list[Path]:
    return sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def file_hash(path: Path) -> str:
    """SHA-256 of the raw file bytes — used to detect content changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_document(text: str, source: str, doc_version: int = 1) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    now = datetime.utcnow().isoformat()
    chunks: list[Chunk] = []
    for i, piece in enumerate(splitter.split_text(text)):
        piece = piece.strip()
        if not piece:
            continue
        digest = hashlib.sha1(f"{source}:{i}:{piece}".encode()).hexdigest()[:12]
        cid = f"{Path(source).stem}-{i}-{digest}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=piece,
                source=source,
                metadata={
                    "source": source,
                    "position": i,
                    "doc_version": doc_version,
                    "ingested_at": now,
                },
            )
        )
    return chunks


def load_and_chunk(docs_dir: Path) -> list[Chunk]:
    """Non-versioned: load all docs and chunk. Used for full rebuilds."""
    all_chunks: list[Chunk] = []
    for path in discover(docs_dir):
        text = load_file(path)
        if not text.strip():
            continue
        all_chunks.extend(chunk_document(text, source=str(path)))
    return all_chunks


class VersionedIngester:
    """Tracks content hashes to skip unchanged files and tombstone stale chunks."""

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.manifest_path = manifest_path or config.manifest_path
        self._manifest: dict = self._load()

    def _load(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def ingest(self, docs_dir: Path) -> dict:
        """Incrementally index docs_dir.

        Returns stats: {added, updated, skipped, removed, new_chunks}.
        """
        from ask_my_docs.retrieval import dense, sparse

        current_files = {str(p): p for p in discover(docs_dir)}

        added = updated = skipped = removed = 0
        old_ids_to_delete: list[str] = []
        new_chunks_to_add: list[Chunk] = []

        for source, path in current_files.items():
            fhash = file_hash(path)
            entry = self._manifest.get(source)

            if entry and entry["hash"] == fhash:
                skipped += 1
                continue

            if entry:
                old_ids_to_delete.extend(entry["chunk_ids"])
                updated += 1
            else:
                added += 1

            text = load_file(path)
            if not text.strip():
                continue

            version = (entry["version"] + 1) if entry else 1
            chunks = chunk_document(text, source=source, doc_version=version)
            new_chunks_to_add.extend(chunks)

            self._manifest[source] = {
                "hash": fhash,
                "chunk_ids": [c.chunk_id for c in chunks],
                "version": version,
                "ingested_at": datetime.utcnow().isoformat(),
            }

        # Tombstone sources that no longer exist on disk.
        for source in set(self._manifest) - set(current_files):
            old_ids_to_delete.extend(self._manifest.pop(source)["chunk_ids"])
            removed += 1

        if old_ids_to_delete or new_chunks_to_add:
            dense.update(old_ids_to_delete, new_chunks_to_add)
            sparse.update(old_ids_to_delete, new_chunks_to_add)

        self._save()

        return {
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "removed": removed,
            "new_chunks": len(new_chunks_to_add),
        }
