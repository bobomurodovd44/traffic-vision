"""Turn a per-sample boolean flag series into merged [start, end] segments.

Shared by every rule in src/rules.py: sample a condition at each detector
timestamp, then merge short gaps and drop sub-threshold blips the way the
task README recommends.
"""
from __future__ import annotations


def flags_to_segments(times: list[float], flags: list[bool],
                       min_duration: float = 0.5,
                       merge_gap: float = 1.0) -> list[list[float]]:
    """times/flags: parallel, sorted-by-time. Returns merged [start, end] runs."""
    runs: list[list[float]] = []
    start = None
    prev_t = None
    for t, f in zip(times, flags):
        if f:
            if start is None:
                start = t
            prev_t = t
        elif start is not None:
            runs.append([start, prev_t])
            start = None
    if start is not None:
        runs.append([start, prev_t])
    if not runs:
        return []

    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [[s, e] for s, e in merged if (e - s) >= min_duration]
