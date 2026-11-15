#!/usr/bin/env python3
"""Measure an MLX SAM 2.1 video checkpoint and compare saved masks."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import psutil


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


def load_reference_masks(path: Path | None) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    masks = {}
    for mask_path in sorted(path.glob("*.png")):
        masks[int(mask_path.stem)] = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Optional checkout containing src/mlx_sam; omit to benchmark Sammie's adapter")
    parser.add_argument("--frames", type=Path, default=Path("/tmp/sammie-bench-video-12"))
    parser.add_argument("--model", help="Hugging Face model id when --source is used")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--memory-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--memory-attention-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--num-maskmem", type=int, default=7)
    parser.add_argument("--max-obj-ptrs", type=int, default=16)
    parser.add_argument("--point-x", type=float, default=500.0)
    parser.add_argument("--point-y", type=float, default=600.0)
    parser.add_argument("--cache-limit-mib", type=int, default=256)
    parser.add_argument("--memory-limit-mib", type=int, default=6_144)
    parser.add_argument("--reference-masks", type=Path)
    parser.add_argument("--save-masks-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.source is not None:
        sys.path.insert(0, str(args.source.resolve() / "src"))
        from mlx_sam.video_predictor import SAM2VideoPredictor
        model_name = args.model or "avbiswas/sam2.1-hiera-base-plus-mlx-16bit"
        build_predictor = lambda: SAM2VideoPredictor.from_pretrained(
            model_name,
            image_size=args.image_size,
            memory_dtype=None if args.memory_dtype == "float32" else args.memory_dtype,
            memory_attention_dtype=None if args.memory_attention_dtype == "float32" else args.memory_attention_dtype,
            num_maskmem=args.num_maskmem,
            max_obj_ptrs_in_encoder=args.max_obj_ptrs,
        )
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from sammie.segmentation.mlx_sam21 import build_mlx_sam21_predictor
        model_name = "official SAM 2.1 Base+ / local MLX FP16 conversion"
        build_predictor = lambda: build_mlx_sam21_predictor(
            cache_limit_mib=args.cache_limit_mib,
            memory_limit_mib=args.memory_limit_mib,
        )

    mx.set_cache_limit(args.cache_limit_mib * MIB)
    mx.set_memory_limit(args.memory_limit_mib * MIB)
    mx.reset_peak_memory()
    process = psutil.Process()
    rss_start = process.memory_info().rss
    timings: dict[str, float] = {}
    masks: dict[int, np.ndarray] = {}

    with RssSampler() as rss:
        started = time.perf_counter()
        predictor = build_predictor()
        mx.eval(predictor.model.parameters())
        timings["load_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        state = predictor.init_state(str(args.frames), precompute_image_features=False)
        mx.synchronize()
        timings["initialize_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        _, object_ids, logits = predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=1,
            points=np.array([[args.point_x, args.point_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        mx.synchronize()
        timings["first_click_seconds"] = time.perf_counter() - started
        masks[0] = logits[0, 0] > 0

        started = time.perf_counter()
        for frame_index, _, frame_logits in predictor.propagate_in_video(state):
            masks[frame_index] = frame_logits[0, 0] > 0
        mx.synchronize()
        timings["propagate_seconds"] = time.perf_counter() - started

    references = load_reference_masks(args.reference_masks)
    ious: dict[str, float] = {}
    for frame_index, mask in masks.items():
        reference = references.get(frame_index)
        if reference is None:
            continue
        intersection = np.logical_and(mask, reference).sum()
        union = np.logical_or(mask, reference).sum()
        ious[str(frame_index)] = float(intersection / union) if union else 1.0

    if args.save_masks_dir:
        args.save_masks_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, mask in masks.items():
            cv2.imwrite(str(args.save_masks_dir / f"{frame_index:05d}.png"), mask.astype(np.uint8) * 255)

    frame_count = sum(
        1
        for path in args.frames.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    report = {
        "benchmark": "sam21_mlx_video_probe",
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "input": {
            "frames_dir": str(args.frames.resolve()),
            "frame_count": frame_count,
            "point_xy": [args.point_x, args.point_y],
        },
        "model": {
            "id": model_name,
            "image_size": args.image_size,
            "memory_dtype": args.memory_dtype,
            "memory_attention_dtype": args.memory_attention_dtype,
            "num_maskmem": args.num_maskmem,
            "max_obj_ptrs": args.max_obj_ptrs,
        },
        "timings": {
            **timings,
            "tracked_fps": frame_count / timings["propagate_seconds"],
        },
        "memory": {
            "rss_start_mib": rss_start / MIB,
            "rss_end_mib": process.memory_info().rss / MIB,
            "rss_peak_mib": rss.peak_bytes / MIB,
            "mlx_active_mib": mx.get_active_memory() / MIB,
            "mlx_cache_mib": mx.get_cache_memory() / MIB,
            "mlx_peak_mib": mx.get_peak_memory() / MIB,
            "resident_preprocessed_frames_mib": state["images"].nbytes / MIB,
            "frame_cache_entries": len(getattr(state["images"], "_cache", {})),
        },
        "quality": {
            "reference_frames": len(ious),
            "mean_iou": float(np.mean(list(ious.values()))) if ious else None,
            "minimum_iou": min(ious.values()) if ious else None,
            "per_frame_iou": ious,
        },
        "output": {
            "object_ids": [int(value) for value in object_ids],
            "mask_pixels": {str(k): int(v.sum()) for k, v in masks.items()},
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
