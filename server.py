"""
Docs RAG MCP Server
Indexes a directory of Markdown files into a local ChromaDB vector store
and exposes semantic search to Claude Code via the Model Context Protocol.

Designed as a game design knowledge base for Esforleeg — indexes design docs,
movement system specs, netcode notes, career/AI learning material, and session
notes so they can be queried in natural language during development.

Tools provided:
  - search_docs    Semantic search across all indexed docs
  - list_docs      List all documents currently indexed
  - reindex_docs   Re-index all docs (run after adding/editing files)

Embeddings: nomic-embed-text via Ollama /api/embed (batch endpoint)
Storage:    ~/.local/share/docs-rag/chroma/
Docs path:  DOCS_PATH env var (default: ~/docs/)
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List

import httpx
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from asyncify import aio

# ─── Config ───────────────────────────────────────────────────────────────────

_docs_env = os.environ.get("DOCS_PATH", str(Path.home() / "docs"))
DOCS_PATHS = [Path(p.strip()).resolve() for p in _docs_env.split(",") if p.strip()]
DOCS_PATH  = DOCS_PATHS[0]  # kept for backwards compat with _index_file rel_path
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://anteframe:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DB_PATH     = Path(os.environ.get("DB_PATH", Path.home() / ".local/share/docs-rag/chroma"))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "")  # host:port — use remote HttpClient when set
COLLECTION  = "docs"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
EMBED_BATCH   = int(os.environ.get("EMBED_BATCH", "32"))

PORT = int(os.environ.get("MCP_PORT", "8766"))
mcp = FastMCP("docs-rag", host="0.0.0.0", port=PORT)

# ─── Batch embedder ───────────────────────────────────────────────────────────

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

# ─── ChromaDB setup ───────────────────────────────────────────────────────────

if CHROMA_HOST:
    _host, _port = (CHROMA_HOST.split(":", 1) + ["8000"])[:2]
    client = chromadb.HttpClient(host=_host, port=int(_port))
else:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

_collection_cache = None


def _get_collection():
    """Return a valid collection handle, refreshing if the cached one is stale."""
    global _collection_cache
    if _collection_cache is not None:
        try:
            _collection_cache.count()
            return _collection_cache
        except Exception:
            _collection_cache = None
    _collection_cache = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection_cache

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _doc_id(filepath: str, chunk_idx: int) -> str:
    return hashlib.md5(f"{filepath}:{chunk_idx}".encode()).hexdigest()


def _index_file(path: Path, base: Path = None) -> int:
    """Index a single markdown file. Returns number of chunks added."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return 0

    if not text:
        return 0

    col = _get_collection()
    rel_path = str(path.relative_to(base or DOCS_PATH))
    chunks = _chunk_text(text)

    try:
        existing = col.get(where={"source": rel_path})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    ids       = [_doc_id(rel_path, i) for i in range(len(chunks))]
    documents = chunks
    metadatas = [{"source": rel_path, "chunk": i, "total_chunks": len(chunks)} for i in range(len(chunks))]

    col.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def _ensure_indexed() -> None:
    if _get_collection().count() == 0:
        _run_reindex()


def _run_reindex() -> dict:
    md_files = []
    for dp in DOCS_PATHS:
        if dp.exists():
            md_files.extend(dp.rglob("*.md"))
    total_chunks = 0
    indexed_files = []
    skipped = []

    for f in md_files:
        base = next((dp for dp in DOCS_PATHS if str(f).startswith(str(dp))), DOCS_PATH)
        n = _index_file(f, base)
        if n > 0:
            indexed_files.append(str(f.relative_to(base)))
            total_chunks += n
        else:
            skipped.append(str(f.relative_to(base)))

    pruned = _prune_stale(set(indexed_files) | set(skipped))
    return {
        "files_indexed": len(indexed_files),
        "total_chunks": total_chunks,
        "skipped": skipped,
        "files": indexed_files,
        "pruned_sources": pruned,
        "prune_refused": PRUNE_REFUSED,
    }


# A reindex only (re)adds files it finds, so chunks of a DELETED doc stayed searchable forever (infra-config #1118:
# 137 removed docs still answered queries on CT101). Prune sources with no file behind them. A shared remote store
# (CHROMA_HOST) can hold other indexers' sources, so there it only runs when PRUNE_STALE=1 says so.
PRUNE_STALE = os.environ.get("PRUNE_STALE", "0" if CHROMA_HOST else "1") == "1"
# Floor: a missing/unmounted DOCS_PATH or a half-copied corpus looks exactly like "every doc was deleted". Refuse to
# prune when nothing was found, or when more than this share of indexed sources would go -- a real cleanup that big
# is rare enough to do by hand (PRUNE_MAX_FRACTION=1 for one run).
PRUNE_MAX_FRACTION = float(os.environ.get("PRUNE_MAX_FRACTION", "0.2"))
PRUNE_REFUSED = ""  # last refusal reason; reindex_docs reports it so sync-docs-rag.sh can alert (stderr is CT101-only)


def _refuse(reason: str) -> list:
    global PRUNE_REFUSED
    PRUNE_REFUSED = reason
    print(f"prune refused: {reason}", file=sys.stderr)
    return []


def _prune_stale(current: set) -> list:
    global PRUNE_REFUSED
    PRUNE_REFUSED = ""
    if not PRUNE_STALE:
        return []
    if not current:
        return _refuse("current file set is empty (DOCS_PATH missing/unmounted?)")
    col = _get_collection()
    got = col.get(include=["metadatas"])
    sources = {m.get("source") for m in got["metadatas"] if m}
    stale = sorted(s for s in sources if s not in current)
    if sources and len(stale) > PRUNE_MAX_FRACTION * len(sources):
        return _refuse(f"{len(stale)}/{len(sources)} sources would go (> {PRUNE_MAX_FRACTION:.0%} floor)")
    ids = [i for i, m in zip(got["ids"], got["metadatas"]) if m and m.get("source") in set(stale)]
    for k in range(0, len(ids), 500):
        col.delete(ids=ids[k:k + 500])
    return stale

# ─── MCP Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
@aio
def search_docs(query: str, n_results: int = 5) -> str:
    """
    Semantic search across all indexed docs.
    Returns the most relevant document chunks for the given query.

    Args:
        query:     Natural language query
        n_results: Number of results to return (default 5)
    """
    _ensure_indexed()
    col = _get_collection()

    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"][0]:
        return "No results found."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta["source"]
        chunk  = meta["chunk"] + 1
        total  = meta["total_chunks"]
        score  = round((1 - dist) * 100, 1)
        output.append(f"## [{source}] (chunk {chunk}/{total}, relevance {score}%)\n\n{doc}")

    return "\n\n---\n\n".join(output)


@mcp.tool()
@aio
def list_docs() -> str:
    """List all documents currently indexed in the RAG store."""
    _ensure_indexed()
    col = _get_collection()

    results = col.get(include=["metadatas"])
    if not results["metadatas"]:
        return "No documents indexed yet. Run reindex_docs first."

    sources = sorted(set(m["source"] for m in results["metadatas"]))
    lines = [f"  {s}" for s in sources]
    return f"{len(sources)} documents indexed:\n" + "\n".join(lines)


@mcp.tool()
@aio
def reindex_docs() -> str:
    """
    Re-index all markdown files from the docs directory.
    Run this after adding or editing doc files.
    """
    result = _run_reindex()
    lines = [
        f"Indexed {result['files_indexed']} files ({result['total_chunks']} chunks)",
        f"Docs paths: {', '.join(str(p) for p in DOCS_PATHS)}",
        f"Pruned {len(result['pruned_sources'])} stale sources (file gone)",
    ]
    if result["prune_refused"]:
        lines.append(f"Prune refused: {result['prune_refused']}")
    if result["files"]:
        lines.append("\nFiles indexed:")
        lines.extend(f"  {f}" for f in result["files"])
    if result["skipped"]:
        lines.append("\nSkipped (empty):")
        lines.extend(f"  {f}" for f in result["skipped"])
    return "\n".join(lines)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
