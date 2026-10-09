# process-resource-snapshot

A context manager that captures CPU time, resident-memory high-water mark, and open
file-descriptor count before and after a block, then computes the delta for resource
attribution.

```python
from process_resource_snapshot import ResourceSnapshot, SnapshotDelta
from process_resource_snapshot.core import ResourceSnapshotScope

with ResourceSnapshotScope() as scope:
    sum(range(1000))

d = scope.delta
# d.cpu_delta_seconds : float
# d.max_rss_delta_kb : float
# d.fd_delta         : Optional[int]
```

Exported names:
- `ResourceSnapshot` — a frozen point-in-time sample with fields
  `cpu_seconds`, `max_rss_kb`, `fd_count` (fd_count is `None` off Linux).
- `SnapshotDelta` — a frozen delta with `cpu_delta_seconds`, `max_rss_delta_kb`,
  `fd_delta`.
- `ResourceSnapshotScope` (from `process_resource_snapshot.core`) — the context
  manager. Pass `force_no_fds=True` to skip fd counting on Linux.

`ResourceSnapshot.capture()` and `ResourceSnapshot.capture_no_fds()` are the two
ways to take a standalone sample.

## Why this exists

The problem is attributing a resource change to a specific block of code in a
process-wide view that mixes many callers. A `with` block that brackets the work
and reports only what changed during that span is the smallest useful unit for
that attribution.

The trade-off: we measure `ru_maxrss` (the resident-set high-water mark) rather
than live RSS. Live RSS fluctuates with paging and is reported in different
units across kernels (kB on Linux, bytes on macOS). `ru_maxrss` is monotonic
non-negative within a process lifetime, so `max_rss_delta_kb` has a stable sign
and meaning: it is the amount by which the high-water mark rose during the
block. It will not show memory that was allocated and then freed before the
block ended — that is an accepted limitation.

CPU time is `os.times()` user+system for the calling process only; children are
excluded so attribution stays with the block being measured.

## The awkward edge

`cpu_delta_seconds` can be a small **negative** number. `os.times()` on some
kernels quantises CPU time and two successive samples can appear to go
backwards by a few microseconds. The library does not clamp: callers that
assert `cpu_delta_seconds >= 0` will fail intermittently. Treat it as
"approximately the CPU spent", and compare with a tolerance if you must
threshold.

File-descriptor counting only works on Linux, via `/proc/self/fd`. Off Linux,
`fd_count` is `None` and `fd_delta` is `None`. `None` means "not measured";
zero would mean "measured and found no change", which would be a lie. If your
code under test manipulates `/proc/self/fd` itself, pass `force_no_fds=True`
to avoid distorting the count.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```

Tests that rely on `/proc/self/fd` skip themselves on non-Linux platforms
rather than fail.
