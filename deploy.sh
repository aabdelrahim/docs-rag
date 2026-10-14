#!/usr/bin/env bash
# Deploy docs-rag (MCP server) source -> CT101 /opt/docs-rag. Source-of-truth = this repo; never edit /opt directly.
# CT101 runs only server.py + asyncify.py (unit docs-rag-mcp, DOCS_PATH=/opt/docs/infra-config/docs, its own local
# Chroma store). http_server.py / reindex_standalone.py are for other hosts (ct102 cron) and are not shipped here.
set -euo pipefail
HOST="${DOCS_RAG_HOST:-services}"
DIR=/opt/docs-rag
cd "$(dirname "$0")"

/private/tmp/dragvenv/bin/python test_prune.py >/dev/null 2>&1 || python3 test_prune.py

echo "→ copy server.py asyncify.py requirements.txt → $HOST:$DIR"
for f in server.py asyncify.py requirements.txt; do
  ssh "$HOST" "cat > $DIR/$f.new" < "$f"
  ssh "$HOST" "mv $DIR/$f.new $DIR/$f"
done

echo "→ deps + restart + health"
ssh "$HOST" "set -e; cd $DIR; .venv/bin/pip install -q -r requirements.txt; systemctl restart docs-rag-mcp; sleep 3"
if ! ssh "$HOST" "systemctl is-active --quiet docs-rag-mcp"; then
  echo "❌ docs-rag-mcp NOT active after deploy — last 20 journal lines:" >&2
  ssh "$HOST" "journalctl -u docs-rag-mcp -n 20 --no-pager" >&2 || true
  exit 1
fi
echo "✅ deployed (unit active). Reindex: infra-config scripts/sync-docs-rag.sh"
