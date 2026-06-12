"""
Standalone reindex script for docs-rag.
Indexes all markdown files from DOCS_PATH into ChromaDB using Ollama embeddings.

Usage:
    OLLAMA_URL=http://... DOCS_PATH=/path/to/docs python reindex_standalone.py
"""

import os
import hashlib
from pathlib import Path
from typing import List

import httpx
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

_docs_env  = os.environ.get("DOCS_PATH", str(Path.home() / "docs"))
DOCS_PATHS = [Path(p.strip()).resolve() for p in _docs_env.split(",") if p.strip()]
DOCS_PATH  = DOCS_PATHS[0]
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://anteframe:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DB_PATH     = Path(os.environ.get("DB_PATH", Path.home() / ".local/share/docs-rag/chroma"))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "")  # host:port — use remote HttpClient when set
COLLECTION  = "docs"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
EMBED_BATCH   = int(os.environ.get("EMBED_BATCH", "32"))


class BatchOllamaEmbedder(EmbeddingFunction[Documents]):
    """Embedding function using Ollama /api/embed (batch endpoint, ~5-10× faster than serial)."""

    def __init__(self, base_url: str, model: str, batch_size: int = EMBED_BATCH):
        self._base = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size

    @staticmethod
    def name() -> str:
        return "ollama"  # matches persisted collection metadata from OllamaEmbeddingFunction

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for i in range(0, len(input), self._batch_size):
            batch = input[i : i + self._batch_size]
            resp = httpx.post(
                f"{self._base}/api/embed",
                json={"model": self._model, "input": batch},
                timeout=120.0,
            )
            resp.raise_for_status()
            embeddings.extend(resp.json()["embeddings"])
        return embeddings


embed_fn = BatchOllamaEmbedder(base_url=OLLAMA_URL, model=EMBED_MODEL)

if CHROMA_HOST:
    _host, _port = (CHROMA_HOST.split(":", 1) + ["8000"])[:2]
    client = chromadb.HttpClient(host=_host, port=int(_port))
else:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},
)


def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _doc_id(filepath: str, chunk_idx: int) -> str:
    return hashlib.md5(f"{filepath}:{chunk_idx}".encode()).hexdigest()


def _index_file(path: Path, base: Path = None) -> int:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return 0
    if not text:
        return 0

    rel_path = str(path.relative_to(base or DOCS_PATH))
    chunks = _chunk_text(text)

    try:
        existing = collection.get(where={"source": rel_path})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids       = [_doc_id(rel_path, i) for i in range(len(chunks))]
    metadatas = [{"source": rel_path, "chunk": i, "total_chunks": len(chunks)} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def main():
    print(f"Docs paths:  {', '.join(str(p) for p in DOCS_PATHS)}")
    print(f"Ollama URL:  {OLLAMA_URL}")
    print(f"Embed model: {EMBED_MODEL}  (batch_size={EMBED_BATCH})")
    if CHROMA_HOST:
        print(f"Chroma:      http://{CHROMA_HOST} (remote)")
    else:
        print(f"Chroma DB:   {DB_PATH} (local)")
    print()

    md_files = []
    for dp in DOCS_PATHS:
        if dp.exists():
            md_files.extend(sorted(dp.rglob("*.md")))
        else:
            print(f"  [warn] path not found: {dp}")

    if not md_files:
        print("No .md files found.")
        return

    print(f"Found {len(md_files)} markdown files. Indexing...")
    total_chunks = 0
    indexed, skipped = [], []

    for f in md_files:
        base = next((dp for dp in DOCS_PATHS if str(f).startswith(str(dp))), DOCS_PATH)
        rel = str(f.relative_to(base))
        n = _index_file(f, base)
        if n > 0:
            print(f"  [ok] {rel} ({n} chunks)")
            indexed.append(rel)
            total_chunks += n
        else:
            print(f"  [skip] {rel} (empty)")
            skipped.append(rel)

    print()
    print(f"Done. {len(indexed)} files indexed, {total_chunks} total chunks.")
    if skipped:
        print(f"Skipped (empty): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
