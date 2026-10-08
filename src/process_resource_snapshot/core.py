"""Context manager that captures process resource counters before and after a block.

Exposes three independent resource axes:
- CPU time (user + system), via os.times()
- Open file descriptors, via a count of os.listdir('/proc/self/fd') on Linux or
  a directory walk over a best-effort directory on other platforms.
- Resident memory blocks, via resource.getrusage(RUSAGE_SELF).ru_maxrss.

Design decisions worth stating plainly:
- We measure ru_maxrss, not a live RSS sample. ru_maxrss is the high-water mark of
  resident memory for this process; it never decreases during a process's lifetime,
  so the delta is monotonic non-negative and therefore deterministic in sign. A live
  RSS delta would fluctuate with paging and is reported differently across kernels.
- CPU time is the sum of user and system time for the calling process (children are
  excluded) so that the snapshot attributes work only to the block being measured.
- fd counting is only supported on Linux, where /proc/self/fd is reliable and cheap.
  On non-Linux platforms fd_delta is None rather than zero: None means "not measured",
  zero would mean "we measured and found no change", which would be a lie.
"""

from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

_LINUX = sys.platform.startswith("linux")
_FD_DIR = "/proc/self/fd"


def _count_fds() -> int:
    """Return the number of currently open file descriptors for this process.

    Only meaningful on Linux, where /proc/self/fd exists. The directory itself is
    held open by listdir while we count, which would add one to the count; we
    subtract that one so the number reflects descriptors that were open before
    the call. This keeps a snapshot taken immediately after a previous snapshot
    stable instead of drifting by +1 each time.
    """
    try:
        entries = os.listdir(_FD_DIR)
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        # /proc not mounted (some sandboxes, WSL1 quirks) or not visible.
        # Caller treats None as "unsupported"; raise so the caller can mark
        # the axis as unsupported rather than silently reporting zero.
        raise
    # listdir opens its own fd on _FD_DIR while iterating; subtract it.
    return len(entries) - 1


def _cpu_time(times: os.times_result) -> float:
    """Sum of user and system CPU time for the calling process, excluding children."""
    return float(times[0] + times[1])


@dataclass(frozen=True)
class ResourceSnapshot:
    """A point-in-time sample of the three resource axes.

    fd_count is None on platforms where /proc/self/fd is not available.
    """

    cpu_seconds: float
    max_rss_kb: float
    fd_count: Optional[int]

    @classmethod
    def capture(cls) -> "ResourceSnapshot":
        """Take a snapshot right now.

        Raises if fd counting is attempted on a platform that lacks /proc/self/fd
        AND the caller has not opted out — see ResourceSnapshot.capture_no_fds()
        for the platform-portable variant that sets fd_count to None instead.
        """
        t = os.times()
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        fd: Optional[int]
        if _LINUX:
            try:
                fd = _count_fds()
            except (FileNotFoundError, PermissionError, NotADirectoryError):
                fd = None
        else:
            fd = None
        return cls(cpu_seconds=_cpu_time(t), max_rss_kb=rss, fd_count=fd)

    @classmethod
    def capture_no_fds(cls) -> "ResourceSnapshot":
        """Snapshot with fd_count forced to None.

        Useful when the caller knows fd counting is unavailable or unwanted and
        wants a uniform SnapshotDelta with fd_delta=None rather than a mix.
        """
        t = os.times()
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return cls(cpu_seconds=_cpu_time(t), max_rss_kb=rss, fd_count=None)

    def delta_to(self, other: "ResourceSnapshot") -> "SnapshotDelta":
        """Compute other - self.

        If either snapshot has fd_count=None, fd_delta is None (the axis is
        treated as not measured for the whole interval). cpu_delta can be a
        very small negative due to os.times() rounding on some kernels; we do
        not clamp, because clamping would hide real regressions where the
        clock appears to go backwards.
        """
        cpu_delta = other.cpu_seconds - self.cpu_seconds
        rss_delta = other.max_rss_kb - self.max_rss_kb
        if self.fd_count is None or other.fd_count is None:
            fd_delta: Optional[int] = None
        else:
            fd_delta = other.fd_count - self.fd_count
        return SnapshotDelta(
            cpu_delta_seconds=cpu_delta,
            max_rss_delta_kb=rss_delta,
            fd_delta=fd_delta,
        )


@dataclass(frozen=True)
class SnapshotDelta:
    """Difference between two ResourceSnapshots.

    cpu_delta_seconds: change in user+system CPU time. May be a tiny negative
        if the kernel's reported CPU time quantised downwards between samples;
        do not assume >= 0 in callers.
    max_rss_delta_kb: change in ru_maxrss. Monotonic non-negative within a
        single process lifetime, since ru_maxrss is a high-water mark.
    fd_delta: change in open fd count, or None if fd counting was not
        available for the whole interval.
    """

    cpu_delta_seconds: float
    max_rss_delta_kb: float
    fd_delta: Optional[int]


class ResourceSnapshotScope:
    """Context manager that records before/after snapshots and the delta.

    The delta is available as `.delta` after the block exits. If the block
    raises, the after-snapshot is still taken (with whatever fds were open at
    the point of the exception) so that attribution of leaked fds is possible;
    the exception is then re-raised unchanged.

    Example:
        with ResourceSnapshotScope() as scope:
            do_work()
        print(scope.delta.cpu_delta_seconds)
    """

    __slots__ = ("_before", "_after", "_delta", "_force_no_fds")

    def __init__(self, force_no_fds: bool = False) -> None:
        # force_no_fds lets callers on Linux opt out of fd counting (e.g. when
        # the code under test is known to manipulate /proc/self/fd itself and
        # would distort the count). It does not enable fd counting elsewhere.
        self._before: Optional[ResourceSnapshot] = None
        self._after: Optional[ResourceSnapshot] = None
        self._delta: Optional[SnapshotDelta] = None
        self._force_no_fds = force_no_fds

    def __enter__(self) -> "ResourceSnapshotScope":
        if self._force_no_fds:
            self._before = ResourceSnapshot.capture_no_fds()
        else:
            self._before = ResourceSnapshot.capture()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Take the after-snapshot regardless of whether the block raised, so
        # leaked-fd attribution still works on error paths. We do not swallow
        # the exception: returning None (falsy) propagates it unchanged.
        if self._force_no_fds:
            self._after = ResourceSnapshot.capture_no_fds()
        else:
            self._after = ResourceSnapshot.capture()
        assert self._before is not None  # __enter__ ran
        self._delta = self._before.delta_to(self._after)
        return None

    @property
    def before(self) -> ResourceSnapshot:
        if self._before is None:
            raise RuntimeError("snapshot not started; use 'with ResourceSnapshotScope():'")
        return self._before

    @property
    def after(self) -> ResourceSnapshot:
        if self._after is None:
            raise RuntimeError("snapshot not finished; exit the 'with' block first")
        return self._after

    @property
    def delta(self) -> SnapshotDelta:
        if self._delta is None:
            raise RuntimeError("snapshot not finished; exit the 'with' block first")
        return self._delta


# Public alias so callers can write `from process_resource_snapshot import ResourceSnapshotScope`
# without reaching into internals. Kept as a separate name from ResourceSnapshot so the
# point-in-time sample and the span context manager are not conflated.
ResourceSnapshotScope.__module__ = __name__
