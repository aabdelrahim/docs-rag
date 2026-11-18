"""Unit test: reindex prunes chunks whose source file is gone (infra-config #1118). Run: python test_prune.py"""
import unittest

import server


class FakeCol:
    def __init__(self, rows):
        self.rows = dict(rows)  # id -> source

    def get(self, include=None, where=None):
        ids = list(self.rows)
        return {"ids": ids, "metadatas": [{"source": self.rows[i]} for i in ids]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i)


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.col = FakeCol({"a0": "keep.md", "a1": "keep.md", "g0": "gone.md", "g1": "gone.md", "e0": "empty.md"})
        self._get, server._get_collection = server._get_collection, lambda: self.col
        self._flag, self._frac, self._run = server.PRUNE_STALE, server.PRUNE_MAX_FRACTION, server._run_reindex
        server.PRUNE_MAX_FRACTION = 0.5  # fixture: 1 of 3 sources goes; the floor has its own tests

    def tearDown(self):
        server._get_collection, server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = self._get, self._flag, self._frac
        server._run_reindex = self._run

    def test_prunes_only_missing_sources(self):
        server.PRUNE_STALE = True
        self.assertEqual(server._prune_stale({"keep.md", "empty.md"}), (["gone.md"], ""))
        self.assertEqual(set(self.col.rows.values()), {"keep.md", "empty.md"})

    def test_refuses_empty_current_set(self):
        server.PRUNE_STALE = True
        pruned, why = server._prune_stale(set())
        self.assertEqual((pruned, len(self.col.rows)), ([], 5))
        self.assertIn("empty", why)

    def test_refuses_over_floor(self):
        server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = True, 0.2
        # 2 of 3 sources would go (67 % > 20 %): nothing deleted
        pruned, why = server._prune_stale({"keep.md"})
        self.assertEqual((pruned, len(self.col.rows)), ([], 5))
        self.assertIn("2/3", why)

    def test_prunes_at_floor(self):
        server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = True, 0.2
        self.col.rows.update({f"k{i}": f"k{i}.md" for i in range(7)})  # 10 sources; 2 missing = 20 %, allowed
        current = {"keep.md"} | {f"k{i}.md" for i in range(7)}
        self.assertEqual(server._prune_stale(current), (["empty.md", "gone.md"], ""))

    def test_off_when_disabled(self):
        server.PRUNE_STALE = False
        self.assertEqual(server._prune_stale({"keep.md"}), ([], ""))
        self.assertEqual(len(self.col.rows), 5)

    def test_reindex_output_reports_refusal(self):
        # infra-config scripts/sync-docs-rag.sh greps reindex_docs output for '^Prune refused' to alert
        base = {"files_indexed": 0, "files_changed": 0, "seconds": 0.0, "failed": [], "total_chunks": 0, "skipped": [],
                "files": [], "pruned_sources": []}
        server._run_reindex = lambda: dict(base, prune_refused="3/4 sources would go")
        out = server.reindex_docs.__wrapped__()
        self.assertIn("\nPrune refused: 3/4 sources would go", out)
        server._run_reindex = lambda: dict(base, prune_refused="")
        self.assertNotIn("Prune refused", server.reindex_docs.__wrapped__())


class FakeStore:
    """Chroma stand-in for _index_file_ex: id -> (doc, meta); upsert may be made to fail."""
    def __init__(self):
        self.rows, self.fail_upsert, self.upserts, self.fail_sources = {}, False, 0, set()

    def get(self, where=None, include=None):
        ids = [i for i, (_, m) in self.rows.items() if not where or m["source"] == where["source"]]
        return {"ids": ids, "metadatas": [self.rows[i][1] for i in ids]}

    def upsert(self, ids, documents, metadatas):
        if self.fail_upsert or any(m["source"] in self.fail_sources for m in metadatas):
            raise RuntimeError("timed out in add")
        self.upserts += 1
        for i, d, m in zip(ids, documents, metadatas):
            self.rows[i] = (d, m)

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i)


class IncrementalTest(unittest.TestCase):
    def setUp(self):
        import pathlib, tempfile
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.store = FakeStore()
        self._get, server._get_collection = server._get_collection, lambda: self.store
        self._paths = server.DOCS_PATHS, server.DOCS_PATH, server.PRUNE_STALE
        self._frac2 = server.PRUNE_MAX_FRACTION
        server.DOCS_PATHS, server.DOCS_PATH, server.PRUNE_STALE = [self.dir], self.dir, False

    def tearDown(self):
        import shutil
        server._get_collection = self._get
        server.DOCS_PATHS, server.DOCS_PATH, server.PRUNE_STALE = self._paths
        server.PRUNE_MAX_FRACTION = self._frac2
        shutil.rmtree(self.dir)

    def test_unchanged_file_not_re_embedded(self):
        (self.dir / "a.md").write_text("hello " * 400)  # > 1 chunk
        r1 = server._run_reindex()
        self.assertEqual((r1["files_changed"], self.store.upserts), (1, 1))
        r2 = server._run_reindex()
        self.assertEqual((r2["files_changed"], r2["files_indexed"], self.store.upserts), (0, 1, 1))

    def test_edit_reembeds_and_drops_leftover_chunks(self):
        p = self.dir / "a.md"
        p.write_text("x" * 2000)
        server._run_reindex()
        before = len(self.store.rows)
        p.write_text("y" * 100)  # shrinks to 1 chunk
        r = server._run_reindex()
        self.assertEqual((r["files_changed"], len(self.store.rows)), (1, 1))
        self.assertGreater(before, 1)

    def test_failed_embed_keeps_old_chunks_and_run_continues(self):
        (self.dir / "a.md").write_text("x" * 2000)
        server._run_reindex()
        old = dict(self.store.rows)
        (self.dir / "a.md").write_text("z" * 2000)
        (self.dir / "b.md").write_text("ok")
        self.store.fail_sources = {"a.md"}
        r = server._run_reindex()
        self.assertEqual(r["failed"], ["a.md"])
        self.assertTrue(all(old[i] == self.store.rows[i] for i in old))  # a.md old chunks intact
        self.assertIn("b.md", {m["source"] for _, m in self.store.rows.values()})  # run continued past the failure

    def test_failed_file_not_pruned(self):
        server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = True, 1.0
        (self.dir / "a.md").write_text("x" * 50)
        (self.dir / "b.md").write_text("y" * 50)
        server._run_reindex()
        (self.dir / "a.md").write_text("changed")
        self.store.fail_sources = {"a.md"}
        r = server._run_reindex()
        self.assertEqual((r["failed"], r["pruned_sources"], len(self.store.rows)), (["a.md"], [], 2))

    def test_aborts_after_consecutive_failures(self):
        for n in "abcdefgh":
            (self.dir / f"{n}.md").write_text(n)
        self.store.fail_upsert = True
        old, server.MAX_CONSEC_FAIL = server.MAX_CONSEC_FAIL, 3
        try:
            r = server._run_reindex()
        finally:
            server.MAX_CONSEC_FAIL = old
        self.assertEqual(len(r["failed"]), 8)  # 3 attempted + 5 reported not attempted
        self.assertEqual(self.store.upserts, 0)

    def test_embed_retry_and_4xx_fast_fail(self):
        import httpx
        calls = []
        class R:
            def __init__(s, code): s.status_code = code; s.request = httpx.Request("POST", "http://x")
            def raise_for_status(s):
                if s.status_code >= 400:
                    raise httpx.HTTPStatusError("e", request=s.request, response=s)
            def json(s): return {"embeddings": [[1.0]]}
        seq = [httpx.ReadTimeout("t"), R(503), R(200)]
        def fake_post(*a, **k):
            calls.append(1)
            x = seq.pop(0)
            if isinstance(x, Exception): raise x
            return x
        old = server.httpx.post, server._BACKOFF
        server.httpx.post, server._BACKOFF = fake_post, 0
        try:
            self.assertEqual(server.embed_fn._embed_batch(["a"]), [[1.0]])
            self.assertEqual(len(calls), 3)
            calls.clear(); seq[:] = [R(404)]
            with self.assertRaises(httpx.HTTPStatusError):
                server.embed_fn._embed_batch(["a"])
            self.assertEqual(len(calls), 1)  # 4xx: no retry
        finally:
            server.httpx.post, server._BACKOFF = old


if __name__ == "__main__":
    unittest.main()
