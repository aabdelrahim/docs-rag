"""
Docs RAG MCP Server
Indexes a directory of Markdown files into a local ChromaDB vector store
and exposes semantic search to Claude Code via the Model Context Protocol.

Designed as a game design knowledge base for Esforleeg — indexes design docs,
movement system specs, netcode notes, career/AI learning material, and session
notes so they can be queried in natural language during development.

Tools provided:
  - search_docs    Semantic search across all indexed docs
  - list_docs      List all indexed documents
  - reindex_docs   Re-index all docs (run after adding/editing files)

Embeddings: nomic-embed-text via Ollama (OLLAMA_URL env var)
Storage:    ~/.local/share/docs-rag/chroma/
Docs path:  DOCS_PATH env var (default: ~/docs/)
"""

import os
import hashlib
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ─── Config ───────────────────────────────────────────────────────────────────

_docs_env = os.environ.get("DOCS_PATH", str(Path.home() / "docs"))
DOCS_PATHS = [Path(p.strip()).resolve() for p in _docs_env.split(",") if p.strip()]
DOCS_PATH  = DOCS_PATHS[0]  # kept for backwards compat with _index_file rel_path
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://anteframe:11434")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DB_PATH       = Path(os.environ.get("DB_PATH", Path.home() / ".local/share/docs-rag/chroma"))
COLLECTION    = "docs"
CHUNK_SIZE    = 800   # characters
CHUNK_OVERLAP = 100

DB_PATH.mkdir(parents=True, exist_ok=True)

PORT = int(os.environ.get("MCP_PORT", "8766"))
mcp = FastMCP("docs-rag", host="0.0.0.0", port=PORT)

# ─── ChromaDB setup ───────────────────────────────────────────────────────────

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

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
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

    rel_path = str(path.relative_to(base or DOCS_PATH))
    chunks = _chunk_text(text)

    # Remove existing chunks for this file
    try:
        existing = collection.get(where={"source": rel_path})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids       = [_doc_id(rel_path, i) for i in range(len(chunks))]
    documents = chunks
    metadatas = [{"source": rel_path, "chunk": i, "total_chunks": len(chunks)} for i in range(len(chunks))]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def _ensure_indexed() -> None:
    """Index all docs if the collection is empty."""
    if collection.count() == 0:
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
        # find which base path this file belongs to for rel_path display
        base = next((dp for dp in DOCS_PATHS if str(f).startswith(str(dp))), DOCS_PATH)
        n = _index_file(f, base)
        if n > 0:
            indexed_files.append(str(f.relative_to(base)))
            total_chunks += n
        else:
            skipped.append(str(f.relative_to(base)))

    return {
        "files_indexed": len(indexed_files),
        "total_chunks": total_chunks,
        "skipped": skipped,
        "files": indexed_files,
    }

# ─── MCP Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def search_docs(query: str, n_results: int = 5) -> str:
    """
    Semantic search across all indexed docs.
    Returns the most relevant document chunks for the given query.

    Args:
        query:     Natural language query
        n_results: Number of results to return (default 5)
    """
    _ensure_indexed()

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
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
def list_docs() -> str:
    """List all documents currently indexed in the RAG store."""
    _ensure_indexed()

    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return "No documents indexed yet. Run reindex_docs first."

    sources = sorted(set(m["source"] for m in results["metadatas"]))
    lines = [f"  {s}" for s in sources]
    return f"{len(sources)} documents indexed:\n" + "\n".join(lines)


@mcp.tool()
def reindex_docs() -> str:
    """
    Re-index all markdown files from the docs directory.
    Run this after adding or editing doc files.
    """
    result = _run_reindex()
    lines = [
        f"Indexed {result['files_indexed']} files ({result['total_chunks']} chunks)",
        f"Docs paths: {', '.join(str(p) for p in DOCS_PATHS)}",
    ]
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
