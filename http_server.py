"""
http_server.py — HTTP REST wrapper for docs-rag.

Exposes search, list, reindex as plain HTTP endpoints.
Uses same ChromaDB collection as the MCP server (same DB_PATH).

Run: python http_server.py  (uses same env vars as server.py)
Port: HTTP_PORT env var (default 8767)

Cortex proxies /api/docs-search → here.
"""

import hashlib
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ── Config (mirrors server.py) ────────────────────────────────────────────────
_docs_env = os.environ.get("DOCS_PATH", str(Path.home() / "docs"))
DOCS_PATHS = [Path(p.strip()).resolve() for p in _docs_env.split(",") if p.strip()]
DOCS_PATH  = DOCS_PATHS[0]
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://192.168.50.93:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DB_PATH     = Path(os.environ.get("DB_PATH", Path.home() / ".local/share/docs-rag/chroma"))
COLLECTION  = "docs"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

DB_PATH.mkdir(parents=True, exist_ok=True)

embed_fn = OllamaEmbeddingFunction(
    url=f"{OLLAMA_URL}/api/embeddings",
    model_name=EMBED_MODEL,
)
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:min(start + CHUNK_SIZE, len(text))])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def _doc_id(filepath: str, chunk_idx: int) -> str:
    return hashlib.md5(f"{filepath}:{chunk_idx}".encode()).hexdigest()

def _index_file(path: Path, base: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return 0
    if not text:
        return 0
    rel = str(path.relative_to(base))
    chunks = _chunk_text(text)
    try:
        existing = collection.get(where={"source": rel})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass
    collection.add(
        ids=[_doc_id(rel, i) for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": rel, "chunk": i, "total_chunks": len(chunks)} for i in range(len(chunks))],
    )
    return len(chunks)

def _run_reindex() -> dict:
    indexed, skipped, total = [], [], 0
    for dp in DOCS_PATHS:
        if not dp.exists():
            continue
        for f in dp.rglob("*.md"):
            base = next((d for d in DOCS_PATHS if str(f).startswith(str(d))), DOCS_PATH)
            n = _index_file(f, base)
            rel = str(f.relative_to(base))
            (indexed if n > 0 else skipped).append(rel)
            total += n
    return {"files_indexed": len(indexed), "total_chunks": total,
            "files": indexed, "skipped": skipped}

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="docs-rag HTTP", version="1.0")

@app.get("/health")
def health():
    try:
        return {"ok": True, "chunks": collection.count(),
                "docs_paths": [str(p) for p in DOCS_PATHS]}
    except Exception as e:
        return JSONResponse(status_code=503, content={"ok": False, "error": str(e)})

@app.get("/search")
def search(q: str = Query(..., description="Natural language query"),
           n: int = Query(5, description="Number of results")):
    count = collection.count()
    if count == 0:
        return {"results": [], "query": q, "note": "nothing indexed — POST /reindex first"}
    results = collection.query(
        query_texts=[q],
        n_results=min(n, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "source": meta["source"],
            "chunk": meta["chunk"] + 1,
            "total_chunks": meta["total_chunks"],
            "relevance": round((1 - dist) * 100, 1),
            "text": doc,
        })
    return {"results": hits, "query": q, "count": len(hits)}

@app.get("/list")
def list_docs():
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return {"docs": [], "count": 0}
    sources = sorted(set(m["source"] for m in results["metadatas"]))
    return {"docs": sources, "count": len(sources)}

@app.post("/reindex")
def reindex():
    return _run_reindex()

if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PORT", "8767"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
