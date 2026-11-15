"""Small real-model smoke benchmark for separate car/person tracking.

This uses an isolated in-memory predictor and never changes the user's saved
points or masks. Run from the repository root with:

    .uv/uv run --no-sync python benchmarks/benchmark_sam31_two_objects.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sammie.segmentation.mlx_sam31 import build_mlx_sam31_predictor  # noqa: E402
from sammie.performance_trace import (  # noqa: E402
    PERFORMANCE_TRACE,
    format_trace_report,
)


FRAME = 1793
CAR_POINT = (350, 1750)
PERSON_POINT = (2155, 824)


def _mask_metrics(mask: np.ndarray) -> dict:
    binary = np.asarray(mask).squeeze() > 0
    count, _labels = cv2.connectedComponents(binary.astype(np.uint8))
    return {"area": int(binary.sum()), "components": int(count - 1)}


def main() -> int:
    output_dir = ROOT / "tmp" / "sam31_two_object_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_TRACE.clear()
    PERFORMANCE_TRACE.set_enabled(True)
    predictor = build_mlx_sam31_predictor(feature_cache_size=3)
    state = predictor.init_state(ROOT / "temp" / "frames")
    result = {"frame": FRAME, "car_point": CAR_POINT, "person_point": PERSON_POINT}

    start = perf_counter()
    predictor.add_new_points_or_box(
        state, FRAME, 0, np.array([CAR_POINT], np.float32), np.array([1], np.int32)
    )
    result["car_prompt_seconds"] = round(perf_counter() - start, 3)

    start = perf_counter()
    _frame, object_ids, masks = predictor.add_new_points_or_box(
        state, FRAME, 1, np.array([PERSON_POINT], np.float32), np.array([1], np.int32)
    )
    result["person_prompt_seconds"] = round(perf_counter() - start, 3)
    result["object_ids"] = object_ids

    binary = [np.asarray(mask).squeeze() > 0 for mask in masks]
    result["initial"] = [_mask_metrics(mask) for mask in binary]
    union = np.logical_or(binary[0], binary[1]).sum()
    result["initial_iou"] = round(
        float(np.logical_and(binary[0], binary[1]).sum() / max(1, union)), 6
    )
    result["car_seed_inside"] = bool(binary[0][CAR_POINT[1], CAR_POINT[0]])
    result["person_seed_inside"] = bool(binary[1][PERSON_POINT[1], PERSON_POINT[0]])

    start = perf_counter()
    tracked = []
    for frame_idx, ids, frame_masks in predictor.propagate_in_video(
        state, start_frame_idx=FRAME, max_frame_num_to_track=5
    ):
        frame_binary = [np.asarray(mask).squeeze() > 0 for mask in frame_masks]
        frame_union = np.logical_or(frame_binary[0], frame_binary[1]).sum()
        tracked.append({
            "frame": frame_idx,
            "object_ids": ids,
            "objects": [_mask_metrics(mask) for mask in frame_binary],
            "iou": round(float(
                np.logical_and(frame_binary[0], frame_binary[1]).sum()
                / max(1, frame_union)
            ), 6),
        })
    result["six_frame_tracking_seconds"] = round(perf_counter() - start, 3)
    result["tracked"] = tracked

    predictor.release(state)
    snapshot = PERFORMANCE_TRACE.snapshot()
    totals = defaultdict(lambda: {"calls": 0, "wall_seconds": 0.0})
    for event in snapshot["events"]:
        item = totals[event["name"]]
        item["calls"] += 1
        item["wall_seconds"] += event["duration_ns"] / 1_000_000_000
    result["trace_totals"] = {
        name: {"calls": item["calls"], "wall_seconds": round(item["wall_seconds"], 3)}
        for name, item in sorted(
            totals.items(), key=lambda pair: pair[1]["wall_seconds"], reverse=True
        )
    }
    passed = (
        object_ids == [0, 1]
        and result["car_seed_inside"]
        and result["person_seed_inside"]
        and result["initial_iou"] < 0.05
        and all(len(frame["objects"]) == 2 for frame in tracked)
        and all(frame["iou"] < 0.10 for frame in tracked)
        and all(obj["area"] > 20_000 for frame in tracked for obj in frame["objects"])
    )
    result["passed"] = passed
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "trace.txt").write_text(
        format_trace_report(snapshot) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Saved {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
