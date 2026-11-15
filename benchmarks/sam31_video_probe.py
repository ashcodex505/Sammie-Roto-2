#!/usr/bin/env python3
"""Measure exact per-frame SAM 3.1 MLX point tracking on a bounded clip."""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil
from PIL import Image


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("/tmp/sammie-bench-video-12"))
    parser.add_argument("--model", type=Path, default=Path("checkpoints/sam3.1-mlx-bf16"))
    parser.add_argument("--point-x", type=float, default=680.0)
    parser.add_argument("--point-y", type=float, default=500.0)
    parser.add_argument("--cache-limit-mib", type=int, default=512)
    parser.add_argument("--memory-limit-mib", type=int, default=10_240)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-masks-dir", type=Path)
    args = parser.parse_args()

    frame_paths = sorted(args.frames.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No JPEG frames found in {args.frames}")

    mx.set_cache_limit(args.cache_limit_mib * MIB)
    mx.set_memory_limit(args.memory_limit_mib * MIB)
    mx.reset_peak_memory()

    from mlx_vlm.models.sam3_1.generate import (
        _get_backbone_features,
        _masks_to_detection,
    )
    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import load_model

    process = psutil.Process()
    rss_start = process.memory_info().rss
    timings: dict[str, float | list[float]] = {}
    masks: dict[int, np.ndarray] = {}
    scores: dict[int, list[float]] = {}

    with RssSampler() as rss:
        started = time.perf_counter()
        model = load_model(args.model.resolve())
        processor = Sam31Processor.from_pretrained(str(args.model.resolve()))
        mx.eval(model.parameters())
        timings["load_seconds"] = time.perf_counter() - started

        image0 = Image.open(frame_paths[0]).convert("RGB")
        width, height = image0.size
        tracker_size = model.config.tracker_config.image_size
        point_coords = mx.array(
            [[[args.point_x * tracker_size / width, args.point_y * tracker_size / height]]],
            dtype=mx.float32,
        )
        point_labels = mx.array([[1]], dtype=mx.int32)
        state = model.tracker_init_state(1, object_ids=[1])
        per_frame: list[float] = []

        for frame_index, frame_path in enumerate(frame_paths):
            frame = image0 if frame_index == 0 else Image.open(frame_path).convert("RGB")
            started = time.perf_counter()
            pixels = mx.array(processor.preprocess_image(frame)["pixel_values"])
            backbone = _get_backbone_features(model, pixels)
            if frame_index == 0:
                features = model.tracker_frame_features(backbone)
                output = model.tracker_model.track_step(
                    state,
                    frame_idx=0,
                    is_init_cond_frame=True,
                    frame_features=features,
                    point_inputs={
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                    },
                    num_frames=len(frame_paths),
                )
                mx.eval(output)
            else:
                output = model.track_step(
                    state,
                    backbone,
                    frame_index,
                    run_mem_encoder=True,
                )
            detection = _masks_to_detection(output, 1, frame.size)
            mx.synchronize()
            per_frame.append(time.perf_counter() - started)
            masks[frame_index] = detection.masks[0]
            scores[frame_index] = [float(value) for value in detection.scores]

    timings["per_frame_seconds"] = per_frame
    timings["total_track_seconds"] = sum(per_frame)
    timings["tracked_fps"] = len(frame_paths) / sum(per_frame)

    if args.save_masks_dir:
        args.save_masks_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, mask in masks.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                args.save_masks_dir / f"{frame_index:05d}.png"
            )

    report = {
        "benchmark": "sam31_mlx_video_probe",
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "input": {
            "frames_dir": str(args.frames.resolve()),
            "frame_count": len(frame_paths),
            "frame_size": [width, height],
            "point_xy": [args.point_x, args.point_y],
            "internal_resolution": processor.image_size,
            "tracker_coordinate_size": tracker_size,
        },
        "model": {"path": str(args.model.resolve())},
        "limits": {
            "mlx_cache_limit_mib": args.cache_limit_mib,
            "mlx_memory_limit_mib": args.memory_limit_mib,
        },
        "timings": timings,
        "memory": {
            "rss_start_mib": rss_start / MIB,
            "rss_end_mib": process.memory_info().rss / MIB,
            "rss_peak_mib": rss.peak_bytes / MIB,
            "mlx_active_mib": mx.get_active_memory() / MIB,
            "mlx_cache_mib": mx.get_cache_memory() / MIB,
            "mlx_peak_mib": mx.get_peak_memory() / MIB,
        },
        "output": {
            "scores": scores,
            "mask_pixels": {str(k): int(v.sum()) for k, v in masks.items()},
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
