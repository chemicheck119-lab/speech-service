"""Bounded numeric-only runtime memory observation for the Speech API."""

from __future__ import annotations

import os
from pathlib import Path
import resource
import sys
from typing import Any


MAX_COUNTER_CHARACTERS = 32
MAX_RESOURCE_BYTES = 1 << 60


def _read_counter(path: Path, *, allow_max: bool = False) -> int | None:
    try:
        with path.open("r", encoding="ascii") as source:
            raw = source.read(MAX_COUNTER_CHARACTERS + 1)
    except (OSError, UnicodeError):
        return None
    value = raw.strip()
    if len(raw) > MAX_COUNTER_CHARACTERS or (allow_max and value == "max"):
        return None
    if not value.isascii() or not value.isdecimal():
        return None
    parsed = int(value)
    return parsed if 0 <= parsed <= MAX_RESOURCE_BYTES else None


def _process_current_rss_bytes(statm_path: Path) -> int | None:
    try:
        with statm_path.open("r", encoding="ascii") as source:
            raw = source.read(128)
        fields = raw.split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        value = resident_pages * page_size
    except (OSError, UnicodeError, ValueError, IndexError):
        return None
    return value if 0 <= value <= MAX_RESOURCE_BYTES else None


def _process_max_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    if sys.platform != "darwin":
        value *= 1024
    return value if 0 <= value <= MAX_RESOURCE_BYTES else None


def capture_resource_snapshot(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_statm_path: Path = Path("/proc/self/statm"),
) -> dict[str, Any]:
    """Return only numeric memory counters and availability metadata."""

    v2_current = cgroup_root / "memory.current"
    if v2_current.exists():
        cgroup_version = "v2"
        current = _read_counter(v2_current)
        peak = _read_counter(cgroup_root / "memory.peak")
        limit = _read_counter(cgroup_root / "memory.max", allow_max=True)
    else:
        cgroup_version = "v1"
        memory_root = cgroup_root / "memory"
        current = _read_counter(memory_root / "memory.usage_in_bytes")
        peak = _read_counter(memory_root / "memory.max_usage_in_bytes")
        limit = _read_counter(memory_root / "memory.limit_in_bytes")
        if current is None and peak is None and limit is None:
            cgroup_version = "unavailable"

    process_current = _process_current_rss_bytes(proc_statm_path)
    process_peak = _process_max_rss_bytes()
    observed = (current, peak, limit, process_current, process_peak)
    return {
        "resource_observation_available": any(value is not None for value in observed),
        "cgroup_version": cgroup_version,
        "cgroup_memory_current_bytes": current,
        "cgroup_memory_peak_bytes": peak,
        "cgroup_memory_limit_bytes": limit,
        "process_current_rss_bytes": process_current,
        "process_max_rss_bytes": process_peak,
    }


__all__ = ["capture_resource_snapshot"]
