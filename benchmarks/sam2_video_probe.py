#!/usr/bin/env python3
"""Measure Sammie's current SAM 2.1 Base point-and-track path on macOS."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam2.build_sam import build_sam2_video_predictor


MIB = 1024 * 1024


class RssSampler:
    def __init__(self, interval_seconds: float = 0.02):
        self.process = psutil.Process()
        self.interval = interval_seconds
        self.stop = threading.Event()
        self.peak_bytes = self.process.memory_info().rss

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(self.interval):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("/tmp/sammie-bench-video-12"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_base_plus.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--point-x", type=float, default=680.0)
    parser.add_argument("--point-y", type=float, default=500.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-masks-dir", type=Path)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
        help="Autocast model operations to the selected dtype.",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":
        torch.mps.empty_cache()

    process = psutil.Process()
    rss_start = process.memory_info().rss
    timings: dict[str, float] = {}
    frame_times: list[float] = []
    masks: dict[int, np.ndarray] = {}

    with RssSampler() as rss, torch.inference_mode():
        started = time.perf_counter()
        predictor = build_sam2_video_predictor(
            args.config, str(args.checkpoint), device=device
        )
        sync(device)
        timings["load_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        state = predictor.init_state(
            video_path=str(args.frames),
            async_loading_frames=True,
            offload_video_to_cpu=True,
        )
        sync(device)
        timings["initialize_seconds"] = time.perf_counter() - started

        amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(args.amp_dtype)
        amp_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type != "cpu"
            else nullcontext()
        )
        with amp_context:
            started = time.perf_counter()
            _, object_ids, logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[args.point_x, args.point_y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
                clear_old_points=True,
            )
            sync(device)
            timings["first_click_seconds"] = time.perf_counter() - started
            masks[0] = (logits[0] > 0).cpu().numpy().squeeze()

            started_all = time.perf_counter()
            for frame_index, _, frame_logits in predictor.propagate_in_video(state):
                started = time.perf_counter()
                sync(device)
                frame_times.append(time.perf_counter() - started)
                masks[frame_index] = (frame_logits[0] > 0).cpu().numpy().squeeze()
            sync(device)
            timings["propagate_seconds"] = time.perf_counter() - started_all

    mps_current = torch.mps.current_allocated_memory() if device.type == "mps" else 0
    mps_driver = torch.mps.driver_allocated_memory() if device.type == "mps" else 0
    if args.save_masks_dir:
        import cv2

        args.save_masks_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, mask in masks.items():
            cv2.imwrite(
                str(args.save_masks_dir / f"{frame_index:05d}.png"),
                mask.astype(np.uint8) * 255,
            )
    report = {
        "benchmark": "sam2_video_probe",
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "input": {
            "frames_dir": str(args.frames.resolve()),
            "frame_count": len(list(args.frames.glob("*.jpg"))),
            "point_xy": [args.point_x, args.point_y],
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "config": args.config,
            "amp_dtype": args.amp_dtype,
        },
        "timings": {
            **timings,
            "tracked_fps": len(masks) / timings["propagate_seconds"],
        },
        "memory": {
            "rss_start_mib": rss_start / MIB,
            "rss_end_mib": process.memory_info().rss / MIB,
            "rss_peak_mib": rss.peak_bytes / MIB,
            "mps_current_mib": mps_current / MIB,
            "mps_driver_mib": mps_driver / MIB,
        },
        "output": {
            "object_ids": [int(value) for value in object_ids],
            "frames": sorted(int(value) for value in masks),
            "mask_pixels": {str(k): int(v.sum()) for k, v in masks.items()},
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
