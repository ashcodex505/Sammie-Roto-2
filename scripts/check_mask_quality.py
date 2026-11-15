#!/usr/bin/env python3
"""Measure disconnected specks and enclosed holes in saved object masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sammie.core import fill_small_holes, remove_small_dots


def mask_defects(mask: np.ndarray, small_area: int) -> tuple[int, int]:
    """Return counts of small disconnected regions and enclosed holes."""
    binary = (mask > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = sorted(
        (int(area) for area in stats[1:component_count, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    small_components = sum(area <= small_area for area in component_areas[1:])

    background = (binary == 0).astype(np.uint8)
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(
        background, 8
    )
    height, width = binary.shape
    small_holes = 0
    for label in range(1, hole_count):
        x = hole_stats[label, cv2.CC_STAT_LEFT]
        y = hole_stats[label, cv2.CC_STAT_TOP]
        component_width = hole_stats[label, cv2.CC_STAT_WIDTH]
        component_height = hole_stats[label, cv2.CC_STAT_HEIGHT]
        area = hole_stats[label, cv2.CC_STAT_AREA]
        touches_edge = (
            x == 0 or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_edge and area <= small_area:
            small_holes += 1
    return small_components, small_holes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, default=Path("temp/masks"))
    parser.add_argument("--object", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--small-area", type=int, default=1000)
    parser.add_argument("--fill-holes", type=int, default=0)
    parser.add_argument("--remove-dots", type=int, default=0)
    parser.add_argument("--max-small-components", type=int, default=0)
    parser.add_argument("--max-small-holes", type=int, default=0)
    args = parser.parse_args()

    missing = []
    component_total = 0
    hole_total = 0
    component_frames = []
    hole_frames = []
    for frame in range(args.start, args.end + 1):
        path = args.mask_dir / f"{frame:05d}" / f"{args.object}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing.append(frame)
            continue
        if args.fill_holes:
            mask = fill_small_holes(mask, args.fill_holes)
        if args.remove_dots:
            mask = remove_small_dots(mask, args.remove_dots)
        components, holes = mask_defects(mask, args.small_area)
        component_total += components
        hole_total += holes
        if components:
            component_frames.append((frame, components))
        if holes:
            hole_frames.append((frame, holes))

    print(f"frames={args.end - args.start + 1} missing={len(missing)}")
    print(
        f"small_components={component_total} "
        f"affected_frames={len(component_frames)}"
    )
    print(f"small_holes={hole_total} affected_frames={len(hole_frames)}")
    if component_frames:
        print("worst_component_frames=", sorted(
            component_frames, key=lambda item: item[1], reverse=True
        )[:5])
    if hole_frames:
        print("worst_hole_frames=", sorted(
            hole_frames, key=lambda item: item[1], reverse=True
        )[:5])

    failed = bool(missing)
    failed |= component_total > args.max_small_components
    failed |= hole_total > args.max_small_holes
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
