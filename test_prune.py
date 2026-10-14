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
        self._flag = server.PRUNE_STALE

    def tearDown(self):
        server._get_collection, server.PRUNE_STALE = self._get, self._flag

    def test_prunes_only_missing_sources(self):
        server.PRUNE_STALE = True
        self.assertEqual(server._prune_stale({"keep.md", "empty.md"}), ["gone.md"])
        self.assertEqual(set(self.col.rows.values()), {"keep.md", "empty.md"})

    def test_off_when_disabled(self):
        server.PRUNE_STALE = False
        self.assertEqual(server._prune_stale({"keep.md"}), [])
        self.assertEqual(len(self.col.rows), 5)


if __name__ == "__main__":
    unittest.main()
