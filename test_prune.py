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
        self._flag, self._frac = server.PRUNE_STALE, server.PRUNE_MAX_FRACTION
        server.PRUNE_MAX_FRACTION = 0.5  # fixture: 1 of 3 sources goes; the floor has its own tests

    def tearDown(self):
        server._get_collection, server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = self._get, self._flag, self._frac

    def test_prunes_only_missing_sources(self):
        server.PRUNE_STALE = True
        self.assertEqual(server._prune_stale({"keep.md", "empty.md"}), ["gone.md"])
        self.assertEqual(set(self.col.rows.values()), {"keep.md", "empty.md"})

    def test_refuses_empty_current_set(self):
        server.PRUNE_STALE = True
        self.assertEqual(server._prune_stale(set()), [])
        self.assertEqual(len(self.col.rows), 5)
        self.assertIn("empty", server.PRUNE_REFUSED)

    def test_refuses_over_floor(self):
        server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = True, 0.2
        # 2 of 3 sources would go (67 % > 20 %): nothing deleted
        self.assertEqual(server._prune_stale({"keep.md"}), [])
        self.assertEqual(len(self.col.rows), 5)
        self.assertIn("2/3", server.PRUNE_REFUSED)

    def test_prunes_at_floor(self):
        server.PRUNE_STALE, server.PRUNE_MAX_FRACTION = True, 0.2
        self.col.rows.update({f"k{i}": f"k{i}.md" for i in range(7)})  # 10 sources; 2 missing = 20 %, allowed
        current = {"keep.md"} | {f"k{i}.md" for i in range(7)}
        self.assertEqual(server._prune_stale(current), ["empty.md", "gone.md"])
        self.assertEqual(server.PRUNE_REFUSED, "")

    def test_off_when_disabled(self):
        server.PRUNE_STALE = False
        self.assertEqual(server._prune_stale({"keep.md"}), [])
        self.assertEqual(len(self.col.rows), 5)


if __name__ == "__main__":
    unittest.main()
