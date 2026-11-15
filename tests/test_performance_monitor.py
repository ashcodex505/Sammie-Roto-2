import unittest
from unittest.mock import patch

from sammie.performance_monitor import MIB, assess_health, format_bytes
from sammie.performance_trace import PerformanceTraceRecorder, format_trace_report


class PerformanceHealthTests(unittest.TestCase):
    def _metrics(self, **overrides):
        metrics = {
            "pressure_level": 1,
            "available_percent": 40,
            "swap_out_rate": 0,
            "process_memory_ratio": 0.1,
            "process_cpu": 5,
            "operation": "Idle",
        }
        metrics.update(overrides)
        return metrics

    def test_busy_cpu_is_healthy_when_memory_is_healthy(self):
        label, _, _ = assess_health(
            self._metrics(process_cpu=180, operation="Running matting")
        )
        self.assertEqual(label, "Busy — Healthy")

    def test_existing_swap_does_not_make_health_bad(self):
        label, _, _ = assess_health(self._metrics(swap_used=4 * 1024 ** 3))
        self.assertEqual(label, "Good")

    def test_active_swap_writes_raise_warning(self):
        label, _, _ = assess_health(self._metrics(swap_out_rate=70 * MIB))
        self.assertEqual(label, "High Pressure")

    def test_kernel_critical_pressure_wins(self):
        label, _, _ = assess_health(self._metrics(pressure_level=4))
        self.assertEqual(label, "Critical Pressure")

    def test_byte_formatting_is_compact(self):
        self.assertEqual(format_bytes(None), "Not active")
        self.assertEqual(format_bytes(512 * MIB), "512 MB")
        self.assertEqual(format_bytes(2 * 1024 ** 3), "2.00 GB")


class PerformanceTraceTests(unittest.TestCase):
    def test_nested_calls_capture_device_duration_resources_and_details(self):
        recorder = PerformanceTraceRecorder(max_events=10)
        recorder.set_enabled(True)
        snapshots = iter([
            {"cpu_ns": 10, "rss": 100, "mlx_active": 1000, "mlx_cache": 50},
            {"cpu_ns": 20, "rss": 120, "mlx_active": 1100, "mlx_cache": 60},
            {"cpu_ns": 35, "rss": 140, "mlx_active": 1200, "mlx_cache": 70},
            {"cpu_ns": 50, "rss": 160, "mlx_active": 1300, "mlx_cache": 80},
        ])
        times = iter([1_000, 2_000, 4_000, 8_000])
        with patch("sammie.performance_trace._resource_snapshot", side_effect=snapshots), \
                patch("sammie.performance_trace.time.perf_counter_ns", side_effect=times):
            with recorder.span("frame", device="pipeline", frame=7):
                with recorder.span("backbone", device="metal", cache="miss"):
                    pass

        events = recorder.snapshot()["events"]
        self.assertEqual([event["name"] for event in events], ["backbone", "frame"])
        self.assertEqual(events[0]["device"], "metal")
        self.assertEqual(events[0]["duration_ns"], 2_000)
        self.assertEqual(events[0]["cpu_ns"], 15)
        self.assertEqual(events[0]["rss_delta"], 20)
        self.assertEqual(events[0]["mlx_active_delta"], 100)
        self.assertEqual(events[0]["details"], {"cache": "miss"})
        self.assertEqual(events[1]["details"], {"frame": 7})

    def test_report_is_pasteable_and_includes_active_calls(self):
        recorder = PerformanceTraceRecorder(max_events=10)
        recorder.set_enabled(True)
        context = recorder.span("track_step", device="metal", frame=3)
        context.__enter__()
        try:
            report = format_trace_report(recorder.snapshot())
        finally:
            context.__exit__(None, None, None)

        self.assertIn("SAMMIE PERFORMANCE TRACE", report)
        self.assertIn("ACTIVE", report)
        self.assertIn("track_step", report)
        self.assertIn("frame=3", report)


if __name__ == "__main__":
    unittest.main()
