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
        self.rows, self.fail_upsert, self.upserts = {}, False, 0

    def get(self, where=None, include=None):
        ids = [i for i, (_, m) in self.rows.items() if not where or m["source"] == where["source"]]
        return {"ids": ids, "metadatas": [self.rows[i][1] for i in ids]}

    def upsert(self, ids, documents, metadatas):
        if self.fail_upsert:
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
        server.DOCS_PATHS, server.DOCS_PATH, server.PRUNE_STALE = [self.dir], self.dir, False

    def tearDown(self):
        import shutil
        server._get_collection = self._get
        server.DOCS_PATHS, server.DOCS_PATH, server.PRUNE_STALE = self._paths
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
        p = self.dir / "a.md"
        p.write_text("x" * 2000)
        server._run_reindex()
        old = dict(self.store.rows)
        p.write_text("z" * 2000)
        (self.dir / "b.md").write_text("ok")
        self.store.fail_upsert = True
        r = server._run_reindex()
        self.assertEqual(sorted(r["failed"]), ["a.md", "b.md"])  # both fail here, neither aborts the run
        self.assertEqual(self.store.rows, old)  # nothing deleted before the add (old order lost the doc)

    def test_failed_file_not_pruned(self):
        server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = True, 1.0
        (self.dir / "a.md").write_text("x" * 50)
        server._run_reindex()
        self.store.fail_upsert = True
        (self.dir / "a.md").write_text("changed")
        r = server._run_reindex()
        self.assertEqual((r["failed"], r["pruned_sources"]), (["a.md"], []))
        self.assertEqual(len(self.store.rows), 1)


if __name__ == "__main__":
    unittest.main()
