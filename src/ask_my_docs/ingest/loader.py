"""Load documents from disk and split them into chunks with metadata."""
from __future__ import annotations

import hashlib
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


def chunk_document(text: str, source: str) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for i, piece in enumerate(splitter.split_text(text)):
        piece = piece.strip()
        if not piece:
            continue
        # Deterministic id from source + position + content hash.
        digest = hashlib.sha1(f"{source}:{i}:{piece}".encode()).hexdigest()[:12]
        cid = f"{Path(source).stem}-{i}-{digest}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=piece,
                source=source,
                metadata={"source": source, "position": i},
            )
        )
    return chunks


def load_and_chunk(docs_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in discover(docs_dir):
        text = load_file(path)
        if not text.strip():
            continue
        all_chunks.extend(chunk_document(text, source=str(path)))
    return all_chunks
