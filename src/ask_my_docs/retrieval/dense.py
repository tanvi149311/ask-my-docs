"""Dense retrieval backed by a persistent Chroma collection."""
from __future__ import annotations

import chromadb
from langchain_openai import OpenAIEmbeddings

from ask_my_docs.config import config
from ask_my_docs.schema import Chunk, Retrieved


def _embedder() -> OpenAIEmbeddings:
    config.require_openai()
    return OpenAIEmbeddings(model=config.embed_model, api_key=config.openai_api_key)


def _client() -> chromadb.api.ClientAPI:
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.chroma_dir))


def build(chunks: list[Chunk]) -> None:
    """Embed chunks and (re)create the Chroma collection from scratch."""
    client = _client()
    try:
        client.delete_collection(config.collection)
    except Exception:
        pass
    coll = client.create_collection(config.collection)

    embedder = _embedder()
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)
    coll.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )


def update(to_delete: list[str], to_add: list[Chunk]) -> None:
    """Incremental update: remove stale chunk IDs, embed and upsert new chunks."""
    client = _client()
    coll = client.get_or_create_collection(config.collection)

    if to_delete:
        coll.delete(ids=to_delete)

    if to_add:
        embedder = _embedder()
        texts = [c.text for c in to_add]
        vectors = embedder.embed_documents(texts)
        coll.upsert(
            ids=[c.chunk_id for c in to_add],
            embeddings=vectors,
            documents=texts,
            metadatas=[c.metadata for c in to_add],
        )


def search(query: str, k: int | None = None) -> list[Retrieved]:
    k = k or config.dense_k
    client = _client()
    coll = client.get_collection(config.collection)
    qvec = _embedder().embed_query(query)
    res = coll.query(query_embeddings=[qvec], n_results=k)

    out: list[Retrieved] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(ids)])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        chunk = Chunk(chunk_id=cid, text=doc, source=meta.get("source", ""), metadata=meta)
        score = 1.0 / (1.0 + dist) if dist is not None else 0.0
        out.append(Retrieved(chunk=chunk, score=score))
    return out
