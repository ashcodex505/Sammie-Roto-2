"""Lightweight, live performance monitoring for the Sammie UI."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
import time
from typing import Any

import psutil
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from sammie.performance_trace import PERFORMANCE_TRACE, format_trace_report


MIB = 1024 ** 2
GIB = 1024 ** 3


def format_bytes(value: int | float | None) -> str:
    """Format a byte count for a compact UI label."""
    if value is None:
        return "Not active"
    value = max(0.0, float(value))
    if value >= GIB:
        return f"{value / GIB:.2f} GB"
    if value >= MIB:
        return f"{value / MIB:.0f} MB"
    if value >= 1024:
        return f"{value / 1024:.0f} KB"
    return f"{value:.0f} B"


def assess_health(metrics: dict[str, Any]) -> tuple[str, str, str]:
    """Return a user-facing health label, explanation, and display color.

    Existing swap usage is not treated as a problem by itself. macOS may keep
    old pages in swap after pressure has passed, so current pressure and new
    swap writes are much more useful signals.
    """
    pressure = metrics.get("pressure_level")
    available = float(metrics.get("available_percent", 100.0))
    swap_out_rate = float(metrics.get("swap_out_rate", 0.0))
    process_ratio = float(metrics.get("process_memory_ratio", 0.0))
    cpu = float(metrics.get("process_cpu", 0.0))
    operation = metrics.get("operation", "Idle")

    if pressure == 4 or available < 3 or swap_out_rate >= 256 * MIB:
        return (
            "Critical Pressure",
            "Memory is critically tight. Stop the current job or unload a model to avoid heavy SSD swapping.",
            "#d64545",
        )
    if pressure == 2 or available < 8 or swap_out_rate >= 64 * MIB or process_ratio >= 0.70:
        return (
            "High Pressure",
            "Memory is under heavy pressure. Unloading SAM or lowering the processing resolution is recommended.",
            "#e06c3b",
        )
    if available < 15 or swap_out_rate >= 8 * MIB or process_ratio >= 0.50:
        return (
            "Watch Memory",
            "The job can continue, but memory is getting tight. Watch whether swap writes keep increasing.",
            "#d6a52d",
        )
    if operation != "Idle" and cpu >= 50:
        return (
            "Busy — Healthy",
            "Sammie is using the processor, but macOS memory pressure and swap activity are healthy.",
            "#4b9bd8",
        )
    return (
        "Good",
        "Memory pressure is normal and Sammie is not causing significant new swap activity.",
        "#4aa564",
    )


def _macos_pressure_level() -> int | None:
    """Read macOS's kernel memory-pressure state without launching a process."""
    if platform.system() != "Darwin":
        return None
    try:
        value = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        libc = ctypes.CDLL(None)
        result = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
        return value.value if result == 0 else None
    except Exception:
        return None


def _framework_memory() -> dict[str, int | None]:
    """Read already-loaded MLX and PyTorch allocators without loading models."""
    values: dict[str, int | None] = {
        "mlx_active": None,
        "mlx_cache": None,
        "mlx_peak": None,
        "mps_allocated": None,
    }

    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            values["mlx_active"] = int(mx.get_active_memory())
            values["mlx_cache"] = int(mx.get_cache_memory())
            values["mlx_peak"] = int(mx.get_peak_memory())
        except Exception:
            pass

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                values["mps_allocated"] = int(torch.mps.current_allocated_memory())
        except Exception:
            pass
    return values


class PerformanceSampler(QThread):
    """Collect metrics off the GUI thread once per second."""

    sample_ready = Signal(dict)

    def __init__(self, parent=None, interval_seconds: float = 1.0):
        super().__init__(parent)
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._operation_lock = threading.Lock()
        self._operation = "Idle"
        self._operation_started: float | None = None
        self._process = psutil.Process(os.getpid())
        self._last_swap_out: int | None = None
        self._last_sample_time: float | None = None

    def begin_operation(self, name: str) -> None:
        with self._operation_lock:
            self._operation = name
            self._operation_started = time.monotonic()

    def end_operation(self) -> None:
        with self._operation_lock:
            self._operation = "Idle"
            self._operation_started = None

    def stop(self) -> None:
        self._stop_event.set()

    def _sample(self) -> dict[str, Any]:
        now = time.monotonic()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process_memory = self._process.memory_info().rss

        if self._last_swap_out is None or self._last_sample_time is None:
            swap_out_rate = 0.0
        else:
            elapsed = max(now - self._last_sample_time, 0.001)
            swap_out_rate = max(0, swap.sout - self._last_swap_out) / elapsed
        self._last_swap_out = swap.sout
        self._last_sample_time = now

        with self._operation_lock:
            operation = self._operation
            started = self._operation_started

        metrics: dict[str, Any] = {
            "process_memory": process_memory,
            "process_memory_ratio": process_memory / max(virtual.total, 1),
            "process_cpu": self._process.cpu_percent(interval=None),
            "pressure_level": _macos_pressure_level(),
            "available_percent": virtual.available / max(virtual.total, 1) * 100,
            "swap_used": swap.used,
            "swap_total": swap.total,
            "swap_out_rate": swap_out_rate,
            "operation": operation,
            "operation_elapsed": now - started if started is not None else None,
        }
        metrics.update(_framework_memory())
        metrics["trace_snapshot"] = PERFORMANCE_TRACE.snapshot()
        return metrics

    def run(self) -> None:
        self._process.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            try:
                self.sample_ready.emit(self._sample())
            except (psutil.Error, OSError):
                pass
            self._stop_event.wait(self.interval_seconds)


class PerformanceMonitorWindow(QDialog):
    """Compact pop-out window for live performance statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sammie Performance Monitor")
        self.setWindowFlag(Qt.Tool, True)
        self.setMinimumWidth(700)
        self.resize(860, 760)
        self._value_labels: dict[str, QLabel] = {}
        self._last_trace_snapshot: dict[str, Any] = PERFORMANCE_TRACE.snapshot()
        self._build_ui()
        self.sampler = PerformanceSampler(self)
        self.sampler.sample_ready.connect(self.update_metrics)
        self.sampler.start()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.health_label = QLabel("Starting monitor…")
        self.health_label.setAlignment(Qt.AlignCenter)
        self.health_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 10px; border-radius: 6px;"
        )
        layout.addWidget(self.health_label)

        self.health_explanation = QLabel("Collecting the first performance sample.")
        self.health_explanation.setWordWrap(True)
        layout.addWidget(self.health_explanation)

        group = QGroupBox("Live Statistics")
        grid = QGridLayout(group)
        rows = [
            ("operation", "Current operation", "The task Sammie is currently performing and how long it has run."),
            ("current_call", "Current model call", "The deepest instrumented CPU, Metal, transfer, or synchronization call running now."),
            ("process_memory", "Sammie memory", "Physical unified memory currently held by the Sammie process."),
            ("process_cpu", "Sammie CPU", "CPU used by Sammie. More than 100% means it is using more than one CPU core."),
            ("pressure", "System memory pressure", "macOS's overall memory-pressure state. This is more meaningful than RAM usage alone."),
            ("swap", "System swap", "SSD space used as temporary memory. The write rate shows whether new swapping is happening now."),
            ("mlx_active", "MLX active memory", "Memory MLX is actively using for model weights and calculations."),
            ("mlx_cache", "MLX reusable cache", "Memory MLX keeps ready for reuse. It can be released when needed."),
            ("mlx_peak", "MLX peak memory", "Highest MLX allocation reached since Sammie started."),
            ("mps_allocated", "PyTorch MPS allocated", "Memory currently allocated by PyTorch for Apple GPU operations such as matting."),
        ]
        for row, (key, title, tooltip) in enumerate(rows):
            name = QLabel(f"{title}:")
            name.setToolTip(tooltip)
            value = QLabel("—")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setToolTip(tooltip)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._value_labels[key] = value
        grid.setColumnStretch(1, 1)
        layout.addWidget(group)

        note = QLabel(
            "High CPU is normal during tracking, matting, and export. The clearest warning signs are "
            "yellow/red memory pressure and a swap write rate that keeps rising. MLX and MPS values "
            "share the same physical memory on Apple silicon, so do not add them together."
        )
        note.setWordWrap(True)
        note.setStyleSheet("padding: 4px;")
        layout.addWidget(note)

        trace_group = QGroupBox("SAM / MLX Call Trace")
        trace_layout = QVBoxLayout(trace_group)
        trace_note = QLabel(
            "Recording starts when this monitor is first opened and continues if it is hidden or "
            "minimized. The trace separates CPU work, Metal/MLX work, GPU synchronization, "
            "transfers, cache activity, and per-frame pipeline time."
        )
        trace_note.setWordWrap(True)
        trace_layout.addWidget(trace_note)

        self.trace_view = QPlainTextEdit()
        self.trace_view.setReadOnly(True)
        self.trace_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.trace_view.setPlaceholderText("Run SAM tracking with this window open to capture calls.")
        trace_layout.addWidget(self.trace_view, 1)

        trace_buttons = QHBoxLayout()
        self.copy_trace_button = QPushButton("Copy Trace for Debugging")
        self.copy_trace_button.setToolTip(
            "Copy a paste-ready trace containing timings and resource deltas, without media paths."
        )
        self.copy_trace_button.clicked.connect(self.copy_trace)
        trace_buttons.addWidget(self.copy_trace_button)
        self.clear_trace_button = QPushButton("Clear Trace")
        self.clear_trace_button.clicked.connect(self.clear_trace)
        trace_buttons.addWidget(self.clear_trace_button)
        trace_buttons.addStretch()
        trace_layout.addLayout(trace_buttons)
        layout.addWidget(trace_group, 1)
        layout.addStretch()

    def begin_operation(self, name: str) -> None:
        self.sampler.begin_operation(name)

    def end_operation(self) -> None:
        self.sampler.end_operation()

    def shutdown(self) -> None:
        PERFORMANCE_TRACE.set_enabled(False)
        self.sampler.stop()
        self.sampler.wait(2000)

    def showEvent(self, event) -> None:
        PERFORMANCE_TRACE.set_enabled(True)
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        # Keep recording after the user opens the monitor. This lets them hide
        # the floating window, reproduce a slow interaction in the main UI,
        # then reopen it to inspect or copy the trace.
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hide instead of destroying so the button can reopen it quickly."""
        event.ignore()
        self.hide()

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        health, explanation, color = assess_health(metrics)
        self.health_label.setText(health)
        self.health_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 10px; border-radius: 6px; "
            f"background-color: {color}; color: white;"
        )
        self.health_explanation.setText(explanation)

        elapsed = metrics.get("operation_elapsed")
        if elapsed is None:
            operation_text = "Idle"
        else:
            minutes, seconds = divmod(int(elapsed), 60)
            operation_text = f"{metrics['operation']} — {minutes:02d}:{seconds:02d}"
        self._value_labels["operation"].setText(operation_text)
        trace_snapshot = metrics.get("trace_snapshot", {})
        self._last_trace_snapshot = trace_snapshot
        active = trace_snapshot.get("active", [])
        if active:
            current = max(active, key=lambda item: int(item.get("depth", 0)))
            self._value_labels["current_call"].setText(
                f"{current['device']} · {current['name']} — "
                f"{current.get('duration_ns', 0) / 1_000_000:.0f} ms"
            )
        else:
            self._value_labels["current_call"].setText("Idle")
        self._value_labels["process_memory"].setText(format_bytes(metrics["process_memory"]))
        self._value_labels["process_cpu"].setText(f"{metrics['process_cpu']:.0f}%")

        pressure_names = {1: "Normal", 2: "Warning", 4: "Critical"}
        pressure = pressure_names.get(metrics.get("pressure_level"), "Unavailable")
        self._value_labels["pressure"].setText(
            f"{pressure} — {metrics['available_percent']:.0f}% available"
        )
        self._value_labels["swap"].setText(
            f"{format_bytes(metrics['swap_used'])} / {format_bytes(metrics['swap_total'])}"
            f" — writes {format_bytes(metrics['swap_out_rate'])}/s"
        )
        for key in ("mlx_active", "mlx_cache", "mlx_peak", "mps_allocated"):
            self._value_labels[key].setText(format_bytes(metrics.get(key)))

        report = format_trace_report(trace_snapshot, recent=200)
        scroll = self.trace_view.verticalScrollBar()
        was_at_bottom = scroll.value() >= scroll.maximum() - 4
        self.trace_view.setPlainText(report)
        if was_at_bottom:
            scroll.setValue(scroll.maximum())

    def copy_trace(self) -> None:
        report = format_trace_report(self._last_trace_snapshot)
        QApplication.clipboard().setText(report)
        self.copy_trace_button.setText("Copied — Paste It Here")

    def clear_trace(self) -> None:
        PERFORMANCE_TRACE.clear()
        self._last_trace_snapshot = PERFORMANCE_TRACE.snapshot()
        self.trace_view.setPlainText(format_trace_report(self._last_trace_snapshot))
        self.copy_trace_button.setText("Copy Trace for Debugging")
