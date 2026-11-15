#!/usr/bin/env python3
"""Small, reproducible SAM 3.1 MLX inference probe for Apple Silicon.

This deliberately benchmarks one image before the model is connected to Sammie's
GUI.  It records both process RSS (all memory owned by Python) and MLX allocator
memory, because either number by itself can hide the real pressure on unified RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil
from PIL import Image
from safetensors import safe_open


MIB = 1024 * 1024


class RssSampler:
    """Poll peak process RSS without adding heavyweight profiling packages."""

    def __init__(self, interval_seconds: float = 0.02):
        self._process = psutil.Process()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = self._process.memory_info().rss

    def __enter__(self) -> "RssSampler":
        def sample() -> None:
            while not self._stop.wait(self._interval):
                self.peak_bytes = max(
                    self.peak_bytes, self._process.memory_info().rss
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_inventory(model_dir: Path) -> dict[str, object]:
    tensor_counts: Counter[str] = Counter()
    parameter_counts: Counter[str] = Counter()
    total_bytes = 0
    files = sorted(model_dir.glob("*.safetensors"))
    for path in files:
        total_bytes += path.stat().st_size
        with safe_open(path, framework="numpy") as tensors:
            for key in tensors.keys():
                view = tensors.get_slice(key)
                dtype = str(view.get_dtype())
                count = int(np.prod(view.get_shape(), dtype=np.int64))
                tensor_counts[dtype] += 1
                parameter_counts[dtype] += count
    return {
        "files": [path.name for path in files],
        "file_bytes": total_bytes,
        "tensor_counts_by_dtype": dict(tensor_counts),
        "parameter_counts_by_dtype": dict(parameter_counts),
    }


def git_metadata(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/sam3.1-mlx-bf16"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/tmp/sammie-bench-frames/frame_000.png"),
    )
    parser.add_argument("--prompt", default="rabbit")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--warm-runs", type=int, default=2)
    parser.add_argument("--cache-limit-mib", type=int, default=512)
    parser.add_argument("--memory-limit-mib", type=int, default=10_240)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-masks-dir", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_dir = args.model.resolve()
    image_path = args.image.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    if not image_path.is_file():
        raise SystemExit(f"Image does not exist: {image_path}")

    # The cache limit controls reusable allocations. The memory limit is a
    # pressure signal to MLX, not a hard sandbox, so RSS is measured separately.
    mx.set_cache_limit(args.cache_limit_mib * MIB)
    mx.set_memory_limit(args.memory_limit_mib * MIB)
    mx.reset_peak_memory()

    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import load_model

    process = psutil.Process()
    rss_start = process.memory_info().rss
    timings: dict[str, object] = {}

    with RssSampler() as rss:
        started = time.perf_counter()
        model = load_model(model_dir)
        processor = Sam31Processor.from_pretrained(str(model_dir))
        processor.image_size = args.resolution
        predictor = Sam3Predictor(model, processor, score_threshold=args.threshold)
        # Force lazy weight materialization inside the measured load phase.
        mx.eval(model.parameters())
        timings["load_seconds"] = time.perf_counter() - started

        image = Image.open(image_path).convert("RGB")
        run_times: list[float] = []
        result = None
        for _ in range(args.warm_runs + 1):
            started = time.perf_counter()
            result = predictor.predict(image, text_prompt=args.prompt)
            mx.synchronize()
            run_times.append(time.perf_counter() - started)

    assert result is not None
    masks = np.asarray(result.masks, dtype=np.uint8)
    mask_pixels = [int(mask.sum()) for mask in masks]
    mask_hashes = [hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks]
    if args.save_masks_dir:
        args.save_masks_dir.mkdir(parents=True, exist_ok=True)
        for index, mask in enumerate(masks):
            Image.fromarray(mask * 255).save(args.save_masks_dir / f"mask_{index:02d}.png")
    report = {
        "benchmark": "sam31_mlx_image_probe",
        "git": git_metadata(root),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "input": {
            "image": str(image_path),
            "image_sha256": sha256(image_path),
            "image_size": list(image.size),
            "prompt": args.prompt,
            "threshold": args.threshold,
            "internal_resolution": args.resolution,
        },
        "model": {
            "path": str(model_dir),
            **weight_inventory(model_dir),
        },
        "limits": {
            "mlx_cache_limit_mib": args.cache_limit_mib,
            "mlx_memory_limit_mib": args.memory_limit_mib,
        },
        "timings": {
            **timings,
            "cold_inference_seconds": run_times[0],
            "warm_inference_seconds": run_times[1:],
        },
        "memory": {
            "rss_start_mib": rss_start / MIB,
            "rss_end_mib": process.memory_info().rss / MIB,
            "rss_peak_mib": rss.peak_bytes / MIB,
            "mlx_active_mib": mx.get_active_memory() / MIB,
            "mlx_cache_mib": mx.get_cache_memory() / MIB,
            "mlx_peak_mib": mx.get_peak_memory() / MIB,
        },
        "output": {
            "detections": len(result.scores),
            "scores": [float(score) for score in result.scores],
            "boxes_xyxy": np.asarray(result.boxes).tolist(),
            "mask_pixels": mask_pixels,
            "mask_sha256": mask_hashes,
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
