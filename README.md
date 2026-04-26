# docs-rag

A RAG (Retrieval-Augmented Generation) MCP server that indexes a directory of Markdown files into a local ChromaDB vector store and exposes semantic search to Claude Code via the Model Context Protocol.

Built as a game design knowledge base for [Esforleeg](https://github.com/aabdelrahim/esforleeg) — indexes design docs, movement system specs, netcode notes, and session notes so they can be queried in plain English during development without copy-pasting files into context.

## What it demonstrates

- **RAG pipeline** end-to-end: chunking → embedding → vector storage → retrieval
- **MCP tool server** exposing AI capabilities to Claude Code
- **Local-first**: embeddings via [nomic-embed-text](https://ollama.com/library/nomic-embed-text) on Ollama, no external API calls for indexing
- **ChromaDB** for persistent local vector storage with cosine similarity

## Architecture

```
Markdown files (~/infra-config/docs/)
    │
    ▼ chunk (800 chars, 100 overlap)
nomic-embed-text (Ollama on local GPU)
    │
    ▼
ChromaDB (~/.local/share/docs-rag/chroma/)
    │
    ▼ cosine similarity search
MCP tools (search_docs, list_docs, reindex_docs)
    │
    ▼
Claude Code (retrieves grounded answers from design docs)
```

## Tools

| Tool | Description |
|---|---|
| `search_docs` | Semantic search — returns ranked chunks with relevance scores |
| `list_docs` | List all indexed documents |
| `reindex_docs` | Re-index all files (run after edits) |

## Example queries (in Claude Code)

- *"What did I decide about stamina regeneration rate?"*
- *"What are the movement states and transitions?"*
- *"What's the rollback netcode plan for Esforleeg?"*
- *"Which AI portfolio project should I build next?"*

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install mcp chromadb httpx

# Requires Ollama running with nomic-embed-text pulled:
# ollama pull nomic-embed-text
```

Add to `~/.mcp.json`:
```json
{
  "mcpServers": {
    "docs-rag": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/server.py"],
      "env": {
        "DOCS_PATH": "/path/to/your/docs",
        "OLLAMA_URL": "http://localhost:11434",
        "EMBED_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

## Results

Indexes ~15 design documents (~3,500 lines total) into ~180 chunks in under 30 seconds on first run. Subsequent searches return in <100ms locally.
</content>
</invoke>