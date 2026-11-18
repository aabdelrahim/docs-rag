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
import time
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
# #6508: one 120 s Ollama read timeout ("timed out in add") used to abort the whole reindex. Longer per-call
# timeout + retries with backoff; a file that still fails is reported, the rest carry on.
EMBED_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", "300"))
EMBED_RETRIES = max(1, int(os.environ.get("EMBED_RETRIES", "3")))
# after this many files in a row fail, Ollama is down: stop burning retries on the rest (next run resumes: unchanged files skip)
MAX_CONSEC_FAIL = int(os.environ.get("MAX_CONSEC_FAIL", "5"))

PORT = int(os.environ.get("MCP_PORT", "8766"))
mcp = FastMCP("docs-rag", host="0.0.0.0", port=PORT)

# ─── Batch embedder ───────────────────────────────────────────────────────────

_BACKOFF = float(os.environ.get("EMBED_BACKOFF", "5"))


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
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, batch):
        last = None
        for attempt in range(EMBED_RETRIES):
            try:
                resp = httpx.post(
                    f"{self._base}/api/embed",
                    json={"model": self._model, "input": batch},
                    timeout=EMBED_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except httpx.HTTPError as e:  # timeouts, connect errors, 5xx (model reload / af busy)
                last = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise  # 4xx (model missing, bad input) will not recover on retry
                if attempt + 1 < EMBED_RETRIES:
                    time.sleep(_BACKOFF * (2 ** attempt))
        raise last


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
    """Index a single markdown file. Returns number of chunks (0 = empty/unreadable)."""
    return _index_file_ex(path, base)[0]


def _index_file_ex(path: Path, base: Path = None) -> tuple:
    """Index one file incrementally. Returns (chunks, changed).

    Unchanged files (same content hash on every stored chunk, same chunk count) are skipped: no embed calls.
    Changed files are upserted FIRST (deterministic ids), then leftover chunks are deleted, so an embed failure
    mid-file never leaves the doc missing from the store (#6508; the old order deleted before adding).
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return 0, False

    if not text:
        return 0, False

    col = _get_collection()
    rel_path = str(path.relative_to(base or DOCS_PATH))
    chunks = _chunk_text(text)
    digest = hashlib.sha256(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}\n{text}".encode("utf-8")).hexdigest()
    ids = [_doc_id(rel_path, i) for i in range(len(chunks))]

    existing = {"ids": [], "metadatas": []}
    try:
        existing = col.get(where={"source": rel_path}, include=["metadatas"])
    except Exception:
        pass
    if (len(existing["ids"]) == len(chunks) and set(existing["ids"]) == set(ids)
            and all((m or {}).get("hash") == digest for m in existing["metadatas"])):
        return len(chunks), False

    metadatas = [{"source": rel_path, "chunk": i, "total_chunks": len(chunks), "hash": digest}
                 for i in range(len(chunks))]
    col.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    idset = set(ids)
    leftover = [i for i in existing["ids"] if i not in idset]
    for k in range(0, len(leftover), 500):
        col.delete(ids=leftover[k:k + 500])
    return len(chunks), True


def _ensure_indexed() -> None:
    if _get_collection().count() == 0:
        _run_reindex()


def _run_reindex() -> dict:
    t0 = time.time()
    md_files = []
    for dp in DOCS_PATHS:
        if dp.exists():
            md_files.extend(dp.rglob("*.md"))
    total_chunks = 0
    indexed_files = []
    skipped = []
    failed = []
    changed_n = 0
    consec = 0

    for f in md_files:
        base = next((dp for dp in DOCS_PATHS if str(f).startswith(str(dp))), DOCS_PATH)
        rel = str(f.relative_to(base))
        try:
            n, changed = _index_file_ex(f, base)
        except Exception as e:  # one bad file (embed timeout after retries) must not abort the run
            print(f"index failed {rel}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(rel)
            consec += 1
            if consec >= MAX_CONSEC_FAIL:
                rest = [str(g.relative_to(next((dp for dp in DOCS_PATHS if str(g).startswith(str(dp))), DOCS_PATH)))
                        for g in md_files[md_files.index(f) + 1:]]
                print(f"aborting: {consec} files failed in a row; {len(rest)} not attempted", file=sys.stderr)
                failed.extend(rest)
                break
            continue
        consec = 0
        if n > 0:
            indexed_files.append(rel)
            total_chunks += n
            changed_n += changed
        else:
            skipped.append(rel)

    # failed files still exist on disk: they count as current so prune never drops their old chunks
    pruned, refused = _prune_stale(set(indexed_files) | set(skipped) | set(failed))
    return {
        "files_indexed": len(indexed_files),
        "files_changed": changed_n,
        "total_chunks": total_chunks,
        "skipped": skipped,
        "files": indexed_files,
        "failed": failed,
        "seconds": round(time.time() - t0, 1),
        "pruned_sources": pruned,
        "prune_refused": refused,
    }


# A reindex only (re)adds files it finds, so chunks of a DELETED doc stayed searchable forever (infra-config #1118:
# 137 removed docs still answered queries on CT101). Prune sources with no file behind them. A shared remote store
# (CHROMA_HOST) can hold other indexers' sources, so there it only runs when PRUNE_STALE=1 says so.
PRUNE_STALE = os.environ.get("PRUNE_STALE", "0" if CHROMA_HOST else "1") == "1"
# Floor: a missing/unmounted DOCS_PATH or a half-copied corpus looks exactly like "every doc was deleted". Refuse to
# prune when nothing was found, or when more than this share of indexed sources would go -- a real cleanup that big
# is rare enough to do by hand (PRUNE_MAX_FRACTION=1 + restart, for one reindex).
PRUNE_MAX_FRACTION = float(os.environ.get("PRUNE_MAX_FRACTION", "0.2"))


def _prune_stale(current: set) -> tuple:
    """Delete chunks whose source file is gone. Returns (pruned_sources, refusal_reason or "")."""
    if not PRUNE_STALE:
        return [], ""
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
    return stale, ""


def _refuse(reason: str) -> tuple:
    # returned, not kept in a global: reindexes run in worker threads and could overwrite each other's reason.
    # reindex_docs prints it so sync-docs-rag.sh can alert (stderr only reaches CT101's journal).
    print(f"prune refused: {reason}", file=sys.stderr)
    return [], reason

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
        f"Changed {result['files_changed']} files re-embedded, {result['files_indexed'] - result['files_changed']} unchanged, "
        f"{result['seconds']}s",
        f"Docs paths: {', '.join(str(p) for p in DOCS_PATHS)}",
        f"Pruned {len(result['pruned_sources'])} stale sources (file gone)",
    ]
    if result["prune_refused"]:
        lines.append(f"Prune refused: {result['prune_refused']}")
    if result["failed"]:
        lines.append(f"Failed {len(result['failed'])} files (embed error after retries): {', '.join(result['failed'][:20])}")
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
