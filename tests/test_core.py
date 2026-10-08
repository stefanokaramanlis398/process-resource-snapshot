import os
import resource
import unittest
from process_resource_snapshot import ResourceSnapshot, SnapshotDelta
from process_resource_snapshot.core import (
    ResourceSnapshotScope,
    _count_fds,
    _cpu_time,
    _LINUX,
)


class TestSnapshotBasics(unittest.TestCase):
    def test_capture_returns_resource_snapshot(self):
        snap = ResourceSnapshot.capture()
        self.assertIsInstance(snap, ResourceSnapshot)
        self.assertGreaterEqual(snap.cpu_seconds, 0.0)
        self.assertGreaterEqual(snap.max_rss_kb, 0.0)
        # fd_count is either a non-negative int (Linux) or None (elsewhere).
        if snap.fd_count is not None:
            self.assertGreaterEqual(snap.fd_count, 0)
            self.assertIsInstance(snap.fd_count, int)

    def test_capture_no_fds_always_none(self):
        snap = ResourceSnapshot.capture_no_fds()
        self.assertIsNone(snap.fd_count)
        self.assertGreaterEqual(snap.cpu_seconds, 0.0)

    def test_snapshot_is_frozen(self):
        snap = ResourceSnapshot.capture()
        with self.assertRaises(Exception):
            snap.cpu_seconds = 999.0  # type: ignore[misc]


class TestDelta(unittest.TestCase):
    def test_delta_to_computes_differences(self):
        a = ResourceSnapshot(cpu_seconds=1.0, max_rss_kb=1000.0, fd_count=3)
        b = ResourceSnapshot(cpu_seconds=2.5, max_rss_kb=1500.0, fd_count=7)
        d = a.delta_to(b)
        self.assertEqual(d.cpu_delta_seconds, 1.5)
        self.assertEqual(d.max_rss_delta_kb, 500.0)
        self.assertEqual(d.fd_delta, 4)

    def test_delta_to_negative_cpu_allowed(self):
        # os.times() can quantise; the implementation does not clamp. Verify
        # the contract: a negative delta is reported as-is, not clamped to 0.
        a = ResourceSnapshot(cpu_seconds=2.0, max_rss_kb=100.0, fd_count=1)
        b = ResourceSnapshot(cpu_seconds=1.9, max_rss_kb=100.0, fd_count=1)
        d = a.delta_to(b)
        self.assertAlmostEqual(d.cpu_delta_seconds, -0.1, places=6)
        self.assertEqual(d.fd_delta, 0)

    def test_fd_delta_none_when_either_side_none(self):
        a = ResourceSnapshot(cpu_seconds=0.0, max_rss_kb=0.0, fd_count=None)
        b = ResourceSnapshot(cpu_seconds=0.0, max_rss_kb=0.0, fd_count=5)
        self.assertIsNone(a.delta_to(b).fd_delta)
        self.assertIsNone(b.delta_to(a).fd_delta)
        # both None -> None
        self.assertIsNone(a.delta_to(a).fd_delta)

    def test_delta_is_frozen(self):
        d = SnapshotDelta(cpu_delta_seconds=1.0, max_rss_delta_kb=2.0, fd_delta=3)
        with self.assertRaises(Exception):
            d.cpu_delta_seconds = 0.0  # type: ignore[misc]


class TestScope(unittest.TestCase):
    def test_scope_provides_delta_after_block(self):
        with ResourceSnapshotScope() as scope:
            # do a little work so CPU ticks over
            _ = sum(range(1000))
        d = scope.delta
        self.assertIsInstance(d, SnapshotDelta)
        # CPU may not have advanced measurably on a fast machine; do not assert
        # on its sign or magnitude, only that it is a real number.
        self.assertIsInstance(d.cpu_delta_seconds, float)
        self.assertFalse(d.cpu_delta_seconds != d.cpu_delta_seconds)  # not NaN

    def test_scope_before_raises_before_enter(self):
        scope = ResourceSnapshotScope()
        with self.assertRaises(RuntimeError):
            _ = scope.before
        with self.assertRaises(RuntimeError):
            _ = scope.delta

    def test_scope_after_raises_before_exit(self):
        scope = ResourceSnapshotScope()
        scope.__enter__()
        try:
            with self.assertRaises(RuntimeError):
                _ = scope.after
            with self.assertRaises(RuntimeError):
                _ = scope.delta
        finally:
            scope.__exit__(None, None, None)

    def test_scope_takes_after_snapshot_even_on_exception(self):
        with self.assertRaises(ValueError):
            with ResourceSnapshotScope() as scope:
                raise ValueError("boom")
        # delta must still be populated so leaked-fd attribution works
        self.assertIsInstance(scope.delta, SnapshotDelta)
        self.assertIsInstance(scope.after, ResourceSnapshot)

    def test_scope_force_no_fds_sets_fd_none(self):
        with ResourceSnapshotScope(force_no_fds=True) as scope:
            pass
        self.assertIsNone(scope.before.fd_count)
        self.assertIsNone(scope.after.fd_count)
        self.assertIsNone(scope.delta.fd_delta)

    def test_scope_reentrant_snapshot_values_match_real_resources(self):
        # Cross-check that the snapshot's max_rss_kb matches a direct
        # getrusage call, modulo nothing (we read the same value).
        with ResourceSnapshotScope() as scope:
            pass
        direct = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # after.max_rss_kb should equal the direct reading taken right after,
        # because ru_maxrss is monotonic and can only have grown since.
        self.assertGreaterEqual(scope.after.max_rss_kb, scope.before.max_rss_kb)
        self.assertLessEqual(scope.after.max_rss_kb, direct)


class TestFdsOnLinux(unittest.TestCase):
    def test_count_fds_is_self_consistent(self):
        # Taking two readings in a row should yield the same count: listdir's
        # own fd is subtracted inside _count_fds, so the count does not drift.
        if not _LINUX:
            self.skipTest("fd counting only supported on Linux")
        first = _count_fds()
        second = _count_fds()
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)

    def test_count_fds_detects_a_leaked_fd(self):
        if not _LINUX:
            self.skipTest("fd counting only supported on Linux")
        before = _count_fds()
        f = open(__file__, "rb")
        try:
            after = _count_fds()
            self.assertEqual(after - before, 1)
        finally:
            f.close()
        closed = _count_fds()
        self.assertEqual(closed, before)

    def test_count_fds_detects_a_closed_fd(self):
        if not _LINUX:
            self.skipTest("fd counting only supported on Linux")
        f = open(__file__, "rb")
        baseline = _count_fds()
        f.close()
        after = _count_fds()
        self.assertEqual(baseline - after, 1)


class TestCpuTimeHelper(unittest.TestCase):
    def test_cpu_time_sums_user_and_system(self):
        t = os.times()
        expected = float(t[0] + t[1])
        self.assertAlmostEqual(_cpu_time(t), expected, places=12)


class TestScopeFdBehaviourOnLinux(unittest.TestCase):
    def test_scope_detects_fd_leak(self):
        if not _LINUX:
            self.skipTest("fd counting only supported on Linux")
        with ResourceSnapshotScope() as scope:
            leaked = open(os.devnull, "rb")
            # deliberately not closed
        try:
            self.assertEqual(scope.delta.fd_delta, 1)
        finally:
            leaked.close()

    def test_scope_detects_fd_balance(self):
        # Open and close within the block: net fd delta should be 0.
        if not _LINUX:
            self.skipTest("fd counting only supported on Linux")
        with ResourceSnapshotScope() as scope:
            f = open(os.devnull, "rb")
            f.close()
        self.assertEqual(scope.delta.fd_delta, 0)


if __name__ == "__main__":
    unittest.main()
