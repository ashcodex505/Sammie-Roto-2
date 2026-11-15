"""Thread-safe, low-overhead call tracing for performance investigations.

The recorder is deliberately independent of Qt so model code can emit spans
without knowing whether the performance-monitor window is open. Recording is
disabled by default; the monitor enables it while visible.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import os
import platform
import sys
import threading
import time
from typing import Any, Iterator

import psutil


def _mlx_memory() -> tuple[int | None, int | None]:
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return None, None
    try:
        return int(mx.get_active_memory()), int(mx.get_cache_memory())
    except Exception:
        return None, None


_PROCESS = psutil.Process(os.getpid())


def _resource_snapshot() -> dict[str, int | None]:
    """Capture counters that can be safely sampled around an application call."""
    cpu = _PROCESS.cpu_times()
    mlx_active, mlx_cache = _mlx_memory()
    return {
        "cpu_ns": int((cpu.user + cpu.system) * 1_000_000_000),
        "rss": int(_PROCESS.memory_info().rss),
        "mlx_active": mlx_active,
        "mlx_cache": mlx_cache,
    }


def _delta(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return int(after - before)


def _format_bytes_signed(value: int | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value / (1024 ** 2):.1f} MiB"


def _format_details(details: dict[str, Any]) -> str:
    if not details:
        return "-"
    return ",".join(f"{key}={value}" for key, value in sorted(details.items()))


class PerformanceTraceRecorder:
    """Record bounded nested spans from any thread."""

    def __init__(self, max_events: int = 500):
        self.max_events = max(1, int(max_events))
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._active: dict[int, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._local = threading.local()
        self._enabled = False
        self._next_id = 1

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    @contextmanager
    def span(self, name: str, *, device: str = "cpu", **details: Any) -> Iterator[None]:
        if not self.is_enabled():
            yield
            return

        started_ns = time.perf_counter_ns()
        resources_before = _resource_snapshot()
        thread = threading.current_thread()
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            active = {
                "id": event_id,
                "name": str(name),
                "device": str(device),
                "details": dict(details),
                "depth": len(stack),
                "parent_id": stack[-1] if stack else None,
                "thread": thread.name,
                "started_ns": started_ns,
            }
            self._active[event_id] = active
        stack.append(event_id)
        status = "ok"
        try:
            yield
        except BaseException as exc:
            status = f"error:{type(exc).__name__}"
            raise
        finally:
            finished_ns = time.perf_counter_ns()
            resources_after = _resource_snapshot()
            if stack and stack[-1] == event_id:
                stack.pop()
            else:
                try:
                    stack.remove(event_id)
                except ValueError:
                    pass
            event = {
                **active,
                "duration_ns": max(0, finished_ns - started_ns),
                "cpu_ns": max(
                    0,
                    int(resources_after["cpu_ns"] or 0)
                    - int(resources_before["cpu_ns"] or 0),
                ),
                "rss_delta": _delta(resources_after["rss"], resources_before["rss"]),
                "mlx_active_delta": _delta(
                    resources_after["mlx_active"], resources_before["mlx_active"]
                ),
                "mlx_cache_delta": _delta(
                    resources_after["mlx_cache"], resources_before["mlx_cache"]
                ),
                "status": status,
            }
            with self._lock:
                self._active.pop(event_id, None)
                self._events.append(event)

    def snapshot(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        with self._lock:
            events = [dict(event) for event in self._events]
            active = []
            for event in self._active.values():
                item = dict(event)
                item["duration_ns"] = max(0, now_ns - int(item["started_ns"]))
                active.append(item)
        return {"enabled": self.is_enabled(), "events": events, "active": active}


def format_trace_report(snapshot: dict[str, Any], *, recent: int = 500) -> str:
    """Format a compact, paste-ready report without paths or user data."""
    lines = [
        "SAMMIE PERFORMANCE TRACE",
        f"platform={platform.system()} {platform.machine()} python={platform.python_version()}",
        f"recording={snapshot.get('enabled', False)}",
        "state\tid\tdepth\tdevice\twall_ms\tcpu_ms\trss_delta\tmlx_active_delta\t"
        "mlx_cache_delta\tstatus\tthread\tcall\tdetails",
    ]

    def append_event(state: str, event: dict[str, Any]) -> None:
        lines.append(
            "\t".join(
                [
                    state,
                    str(event.get("id", "-")),
                    str(event.get("depth", 0)),
                    str(event.get("device", "-")),
                    f"{float(event.get('duration_ns', 0)) / 1_000_000:.3f}",
                    (
                        f"{float(event['cpu_ns']) / 1_000_000:.3f}"
                        if event.get("cpu_ns") is not None
                        else "n/a"
                    ),
                    _format_bytes_signed(event.get("rss_delta")),
                    _format_bytes_signed(event.get("mlx_active_delta")),
                    _format_bytes_signed(event.get("mlx_cache_delta")),
                    str(event.get("status", "running")),
                    str(event.get("thread", "-")),
                    str(event.get("name", "-")),
                    _format_details(event.get("details", {})),
                ]
            )
        )

    for event in snapshot.get("active", []):
        append_event("ACTIVE", event)
    for event in snapshot.get("events", [])[-max(1, int(recent)):]:
        append_event("DONE", event)
    return "\n".join(lines)


PERFORMANCE_TRACE = PerformanceTraceRecorder()


def trace_performance_call(name: str, *, device: str = "cpu", **details: Any):
    """Convenience entry point used by model adapters."""
    return PERFORMANCE_TRACE.span(name, device=device, **details)
