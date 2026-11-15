"""SAM 3.1 MLX adapter for Sammie's SAM 2-style video predictor contract."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import platform
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from sammie.performance_trace import trace_performance_call


SAM31_MODEL_NAME = "SAM 3.1 MLX"
SAM31_REPO_ID = "mlx-community/sam3.1-bf16"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM31_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam3.1-mlx-bf16"
MIB = 1024 * 1024


def is_mlx_sam31_model(name: str | None) -> bool:
    return name == SAM31_MODEL_NAME


def sam31_checkpoint_ready(path: str | Path = SAM31_CHECKPOINT) -> bool:
    root = Path(path)
    return (root / "config.json").is_file() and (root / "model.safetensors").is_file()


def _load_runtime():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("SAM 3.1 MLX requires an Apple Silicon Mac.")
    try:
        import mlx.core as mx
        from mlx_vlm.models.sam3_1.generate import _get_backbone_features
        from mlx_vlm.models.sam3_1.multiplex import MultiplexState, MultiplexTrackerState
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "SAM 3.1 support is not installed. Run `uv sync --extra sam31`."
        ) from exc
    return mx, _get_backbone_features, MultiplexState, MultiplexTrackerState, Sam31Processor, load_model


def _clone_tracker_state(state, MultiplexState, MultiplexTrackerState):
    """Clone mutable tracking containers while sharing immutable MLX tensors."""
    if state is None:
        return None
    mux = state.multiplex_state
    cloned_mux = MultiplexState(
        [list(bucket) for bucket in mux.assignments],
        allowed_bucket_capacity=mux.allowed_bucket_capacity,
        object_ids=None if mux.object_ids is None else list(mux.object_ids),
    )
    cloned = MultiplexTrackerState(cloned_mux)
    cloned.cond_frame_outputs = _clone_containers(state.cond_frame_outputs)
    cloned.non_cond_frame_outputs = _clone_containers(state.non_cond_frame_outputs)
    return cloned


def _clone_containers(value):
    """Copy mutable structure without duplicating immutable MLX tensor storage."""
    if isinstance(value, dict):
        return {key: _clone_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_containers(item) for item in value)
    if isinstance(value, set):
        return set(value)
    return value


class Sam31MLXPredictor:
    """Translate Sammie's interactive predictor calls to SAM 3.1 tracking."""

    def __init__(self, checkpoint: str | Path = SAM31_CHECKPOINT, *,
                 cache_limit_mib: int = 256, memory_limit_mib: int = 10_240,
                 feature_cache_size: int = 1):
        checkpoint = Path(checkpoint)
        if not sam31_checkpoint_ready(checkpoint):
            raise FileNotFoundError(
                f"SAM 3.1 checkpoint is incomplete: {checkpoint}. "
                f"Download {SAM31_REPO_ID} into that directory."
            )
        with trace_performance_call("sam31.load_runtime", device="cpu"):
            (self.mx, self._get_backbone_features, self._MultiplexState,
             self._MultiplexTrackerState, Sam31Processor, load_model) = _load_runtime()
        with trace_performance_call(
                "sam31.configure_mlx", device="metal",
                cache_limit_mib=cache_limit_mib, memory_limit_mib=memory_limit_mib):
            self.mx.set_cache_limit(max(64, int(cache_limit_mib)) * MIB)
            self.mx.set_memory_limit(max(4096, int(memory_limit_mib)) * MIB)
        with trace_performance_call("sam31.load_model", device="cpu+metal"):
            self.model = load_model(checkpoint.resolve())
        with trace_performance_call("sam31.load_processor", device="cpu"):
            self.processor = Sam31Processor.from_pretrained(str(checkpoint.resolve()))
        with trace_performance_call("sam31.eval_model_state", device="metal.sync"):
            # The Qt loader runs on a worker thread, while prompts and tracking
            # run on the main thread. SAM 3.1 keeps non-parameter MLX arrays in
            # its module state; leaving those lazy makes them retain the worker
            # thread's stream ID and later fail with "There is no Stream ... in
            # current thread." Materialize the complete state before handoff.
            self.mx.eval(self.model.state)
            self.mx.synchronize()
        self.image_size = int(self.model.config.tracker_config.image_size)
        self.feature_cache_size = max(1, int(feature_cache_size))

    def init_state(self, video_path, **_kwargs) -> dict:
        with trace_performance_call("sam31.scan_frames", device="cpu"):
            paths = sorted(path for path in Path(video_path).iterdir()
                           if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            raise ValueError(f"No image frames found in directory: {video_path}")
        with trace_performance_call("sam31.read_frame_metadata", device="cpu", frame=0):
            with Image.open(paths[0]) as image:
                width, height = image.size
        return {
            "images": paths, "num_frames": len(paths), "video_width": width,
            "video_height": height, "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(), "obj_ids": [],
            "point_inputs_per_obj": {}, "mask_inputs_per_obj": {},
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {}, "_tracker_state": None,
            "_feature_cache": OrderedDict(), "_prompt_records": OrderedDict(),
        }

    @staticmethod
    def _empty_output_dict() -> dict:
        return {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    def _register_object(self, state: dict, obj_id: int) -> int:
        existing = state["obj_id_to_idx"].get(obj_id)
        if existing is not None:
            return existing
        idx = len(state["obj_ids"])
        state["obj_id_to_idx"][obj_id] = idx
        state["obj_idx_to_id"][idx] = obj_id
        state["obj_ids"].append(obj_id)
        state["point_inputs_per_obj"][idx] = {}
        state["mask_inputs_per_obj"][idx] = {}
        state["output_dict_per_obj"][idx] = self._empty_output_dict()
        state["temp_output_dict_per_obj"][idx] = self._empty_output_dict()
        state["frames_tracked_per_obj"][idx] = {}
        return idx

    def _frame_features(self, state: dict, frame_idx: int) -> dict:
        cache = state["_feature_cache"]
        cached = cache.pop(frame_idx, None)
        if cached is not None:
            with trace_performance_call(
                    "sam31.feature_cache_hit", device="cache", frame=frame_idx):
                cache[frame_idx] = cached
                return cached
        with trace_performance_call(
                "sam31.frame_features", device="pipeline", frame=frame_idx,
                cache="miss"):
            with trace_performance_call(
                    "sam31.decode_frame", device="cpu", frame=frame_idx):
                with Image.open(state["images"][frame_idx]) as image:
                    rgb_image = image.convert("RGB")
            with trace_performance_call(
                    "sam31.preprocess_image", device="cpu", frame=frame_idx):
                pixel_values = self.processor.preprocess_image(rgb_image)["pixel_values"]
            with trace_performance_call(
                    "sam31.upload_pixels", device="cpu->metal", frame=frame_idx):
                pixels = self.mx.array(pixel_values)
            with trace_performance_call(
                    "sam31.backbone", device="metal", frame=frame_idx):
                backbone = self._get_backbone_features(self.model, pixels)
            with trace_performance_call(
                    "sam31.tracker_frame_features", device="metal", frame=frame_idx):
                features = self.model.tracker_frame_features(backbone)
            with trace_performance_call(
                    "sam31.eval_frame_features", device="metal.sync", frame=frame_idx):
                self.mx.eval(features)
            with trace_performance_call(
                    "sam31.feature_cache_store", device="cache", frame=frame_idx):
                cache[frame_idx] = features
                while len(cache) > self.feature_cache_size:
                    cache.popitem(last=False)
            return features

    def _scaled_points(self, state: dict, points, labels) -> dict:
        coords = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        coords[:, 0] *= self.image_size / state["video_width"]
        coords[:, 1] *= self.image_size / state["video_height"]
        return {
            "point_coords": self.mx.array(coords[None], dtype=self.mx.float32),
            "point_labels": self.mx.array(
                np.asarray(labels, dtype=np.int32).reshape(1, -1), dtype=self.mx.int32),
        }

    def _store_output(self, state: dict, frame_idx: int, output: dict, cond: bool) -> None:
        storage_key = "cond_frame_outputs" if cond else "non_cond_frame_outputs"
        for idx in range(len(state["obj_ids"])):
            state["output_dict_per_obj"][idx][storage_key][frame_idx] = output
            state["frames_tracked_per_obj"][idx][frame_idx] = {"reverse": False}

    def _output_masks(self, state: dict, output: dict) -> list[np.ndarray]:
        with trace_performance_call("sam31.mask_download", device="metal->cpu"):
            masks = np.asarray(output["pred_masks_high_res"])
        if masks.ndim == 3:
            masks = masks[:, None]
        rendered = []
        size = (state["video_width"], state["video_height"])
        for mask in masks:
            plane = np.asarray(mask).squeeze().astype(np.float32, copy=False)
            if plane.shape[::-1] != size:
                with trace_performance_call(
                        "sam31.resize_mask", device="cpu",
                        source=f"{plane.shape[1]}x{plane.shape[0]}",
                        target=f"{size[0]}x{size[1]}"):
                    plane = cv2.resize(plane, size, interpolation=cv2.INTER_LINEAR)
            rendered.append(plane[None])
        return rendered

    @staticmethod
    def _frame_output(tracker_state, frame_idx: int):
        if tracker_state is None:
            return None
        return tracker_state.cond_frame_outputs.get(
            frame_idx, tracker_state.non_cond_frame_outputs.get(frame_idx))

    def _independent_point_mask(self, state: dict, frame_idx: int,
                                point_inputs: dict, obj_id: int):
        with trace_performance_call(
                "sam31.tracker_init_state", device="cpu+metal", objects=1):
            tracker = self.model.tracker_init_state(1, object_ids=[obj_id])
        with trace_performance_call(
                "sam31.track_step_independent", device="metal", frame=frame_idx):
            output = self.model.tracker_model.track_step(
                tracker, frame_idx=frame_idx, is_init_cond_frame=True,
                frame_features=self._frame_features(state, frame_idx),
                point_inputs=point_inputs, num_frames=state["num_frames"])
        with trace_performance_call(
                "sam31.eval_independent", device="metal.sync", frame=frame_idx):
            self.mx.eval(output)
        return output["pred_masks_high_res"]

    def _propagate_one(self, state: dict, frame_idx: int, reverse: bool) -> dict:
        with trace_performance_call(
                "sam31.propagate_frame", device="pipeline", frame=frame_idx,
                reverse=reverse):
            with trace_performance_call(
                    "sam31.track_step", device="metal", frame=frame_idx,
                    reverse=reverse):
                output = self.model.tracker_model.track_step(
                    state["_tracker_state"], frame_idx=frame_idx,
                    is_init_cond_frame=False,
                    frame_features=self._frame_features(state, frame_idx),
                    num_frames=state["num_frames"], track_in_reverse=reverse)
            with trace_performance_call(
                    "sam31.eval_track_step", device="metal.sync", frame=frame_idx):
                self.mx.eval(output)
            with trace_performance_call(
                    "sam31.store_and_prune", device="cpu", frame=frame_idx):
                self._store_output(state, frame_idx, output, cond=False)
                self._prune_non_conditioning_memory(state, frame_idx, reverse)
            return output

    def _prune_non_conditioning_memory(self, state: dict, frame_idx: int,
                                       reverse: bool) -> None:
        """Retain exactly the temporal horizon the model can consult."""
        tracker_model = self.model.tracker_model
        lookback = max(
            int(tracker_model.max_obj_ptrs_in_encoder),
            2 + (int(tracker_model.num_maskmem) - 2)
            * int(tracker_model.memory_temporal_stride_for_eval),
        )
        if reverse:
            keep_min, keep_max = frame_idx, frame_idx + lookback
        else:
            keep_min, keep_max = frame_idx - lookback, frame_idx

        tracker = state["_tracker_state"]
        for old_frame in list(tracker.non_cond_frame_outputs):
            if not (keep_min <= old_frame <= keep_max):
                tracker.non_cond_frame_outputs.pop(old_frame, None)
        # Sammie's compatibility dictionaries must not retain a second copy of
        # every tensor output after the native tracker has evicted it.
        for output_dict in state["output_dict_per_obj"].values():
            non_cond = output_dict["non_cond_frame_outputs"]
            for old_frame in list(non_cond):
                if not (keep_min <= old_frame <= keep_max):
                    non_cond.pop(old_frame, None)

    @staticmethod
    def _discard_non_conditioning_memory(state: dict) -> None:
        """Invalidate propagated outputs after a prompt changes the timeline.

        SAM 3.1 conditioning corrections are authoritative.  Any transient
        frames produced before that correction may describe the old mask (or
        even the old object count), so retaining them makes the next forward
        pass mix two incompatible timelines.
        """
        tracker = state.get("_tracker_state")
        if tracker is not None:
            tracker.non_cond_frame_outputs.clear()
        for collection_name in (
            "output_dict_per_obj", "temp_output_dict_per_obj"
        ):
            for output_dict in state.get(collection_name, {}).values():
                output_dict["non_cond_frame_outputs"].clear()
        for tracked_frames in state.get("frames_tracked_per_obj", {}).values():
            tracked_frames.clear()

    def _ensure_frame(self, state: dict, frame_idx: int) -> dict:
        tracker = state["_tracker_state"]
        existing = self._frame_output(tracker, frame_idx)
        if existing is not None:
            return existing
        known = sorted(set(tracker.cond_frame_outputs) |
                       set(tracker.non_cond_frame_outputs))
        if not known:
            raise RuntimeError("SAM 3.1 has no conditioning frame to propagate from.")
        origin = min(known, key=lambda value: abs(value - frame_idx))
        step = 1 if frame_idx > origin else -1
        output = self._frame_output(tracker, origin)
        for index in range(origin + step, frame_idx + step, step):
            output = self._propagate_one(state, index, reverse=step < 0)
        return output

    def add_new_points_or_box(self, inference_state: dict, frame_idx: int,
                              obj_id: int, points, labels,
                              clear_old_points: bool = True, box=None, **_kwargs):
        if box is not None:
            raise NotImplementedError("SAM 3.1 box prompts are not wired into Sammie yet.")
        state = inference_state
        with trace_performance_call(
                "sam31.prepare_prompt", device="cpu", frame=frame_idx,
                object_id=obj_id, points=len(labels)):
            point_inputs = self._scaled_points(state, points, labels)
        state["_prompt_records"][(int(frame_idx), int(obj_id))] = (
            np.asarray(points, dtype=np.float32).copy(),
            np.asarray(labels, dtype=np.int32).copy())
        obj_idx = state["obj_id_to_idx"].get(obj_id)
        tracker = state["_tracker_state"]

        if tracker is None:
            obj_idx = self._register_object(state, obj_id)
            with trace_performance_call(
                    "sam31.tracker_init_state", device="cpu+metal", objects=1):
                tracker = self.model.tracker_init_state(1, object_ids=[obj_id])
            state["_tracker_state"] = tracker
            with trace_performance_call(
                    "sam31.track_step_prompt", device="metal", frame=frame_idx,
                    mode="first_object"):
                output = self.model.tracker_model.track_step(
                    tracker, frame_idx=frame_idx, is_init_cond_frame=True,
                    frame_features=self._frame_features(state, frame_idx),
                    point_inputs=point_inputs, num_frames=state["num_frames"])
            with trace_performance_call(
                    "sam31.eval_prompt", device="metal.sync", frame=frame_idx):
                self.mx.eval(output)
            self._store_output(state, frame_idx, output, cond=True)
        elif obj_idx is None:
            new_mask = self._independent_point_mask(state, frame_idx, point_inputs, obj_id)
            self._ensure_frame(state, frame_idx)
            with trace_performance_call(
                    "sam31.add_mask_prompt", device="metal", frame=frame_idx,
                    mode="new_object"):
                output = self.model.tracker_model.add_mask_prompt(
                    tracker, frame_idx, self._frame_features(state, frame_idx),
                    new_mask > 0, object_ids=[obj_id])
            with trace_performance_call(
                    "sam31.eval_mask_prompt", device="metal.sync", frame=frame_idx):
                self.mx.eval(output)
            self._discard_non_conditioning_memory(state)
            obj_idx = self._register_object(state, obj_id)
            self._store_output(state, frame_idx, output, cond=True)
        else:
            previous = self._ensure_frame(state, frame_idx)
            count = len(state["obj_ids"])
            point_count = int(point_inputs["point_labels"].shape[1])
            coords_np = np.zeros((count, point_count, 2), dtype=np.float32)
            labels_np = np.full((count, point_count), -1, dtype=np.int32)
            coords_np[obj_idx] = np.asarray(point_inputs["point_coords"])[0]
            labels_np[obj_idx] = np.asarray(point_inputs["point_labels"])[0]
            with trace_performance_call(
                    "sam31.track_step_prompt", device="metal", frame=frame_idx,
                    mode="correction", objects=count):
                output = self.model.tracker_model.track_step(
                    tracker, frame_idx=frame_idx, is_init_cond_frame=True,
                    frame_features=self._frame_features(state, frame_idx),
                    point_inputs={"point_coords": self.mx.array(coords_np),
                                  "point_labels": self.mx.array(labels_np)},
                    prev_sam_mask_logits=previous["pred_masks"],
                    objects_to_interact=list(range(count)),
                    num_frames=state["num_frames"])
            # Corrections are persistent conditioning memories, not transient
            # propagation results.
            tracker.non_cond_frame_outputs.pop(frame_idx, None)
            with trace_performance_call(
                    "sam31.eval_prompt", device="metal.sync", frame=frame_idx):
                self.mx.eval(output)
            self._discard_non_conditioning_memory(state)
            self._store_output(state, frame_idx, output, cond=True)

        state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
        return frame_idx, list(state["obj_ids"]), self._output_masks(state, output)

    def propagate_in_video(self, inference_state: dict, start_frame_idx: int = 0,
                           max_frame_num_to_track: int | None = None,
                           reverse: bool = False, **_kwargs) -> Iterator:
        state = inference_state
        if state["_tracker_state"] is None:
            return
        stop = 0 if reverse else state["num_frames"] - 1
        if max_frame_num_to_track is not None:
            distance = max(0, int(max_frame_num_to_track))
            stop = (max(0, start_frame_idx - distance) if reverse else
                    min(state["num_frames"] - 1, start_frame_idx + distance))
        step = -1 if reverse else 1
        for frame_idx in range(start_frame_idx, stop + step, step):
            with trace_performance_call(
                    "sam31.frame_pipeline", device="pipeline", frame=frame_idx,
                    reverse=reverse):
                output = self._frame_output(state["_tracker_state"], frame_idx)
                if output is None:
                    output = self._propagate_one(state, frame_idx, reverse=reverse)
                masks = self._output_masks(state, output)
            yield frame_idx, list(state["obj_ids"]), masks
            if frame_idx % 30 == 0:
                with trace_performance_call(
                        "sam31.clear_mlx_cache", device="cache", frame=frame_idx):
                    self.mx.clear_cache()

    def clear_frame_if_tracked(self, _state: dict, _obj_id: int,
                               _frame_idx: int) -> bool:
        # Corrections consume the prior mask; do not erase it before replacement points arrive.
        return True

    def snapshot_frame_state(self, state: dict, _obj_id: int, _frame_idx: int) -> dict:
        keys = ("obj_id_to_idx", "obj_idx_to_id", "obj_ids",
                "point_inputs_per_obj", "mask_inputs_per_obj",
                "output_dict_per_obj", "temp_output_dict_per_obj",
                "frames_tracked_per_obj", "_prompt_records")
        snapshot = {"_tracker_state": _clone_tracker_state(
            state["_tracker_state"], self._MultiplexState, self._MultiplexTrackerState)}
        for key in keys:
            snapshot[key] = _clone_containers(state[key])
        return snapshot

    @staticmethod
    def restore_frame_state(state: dict, snapshot: dict) -> None:
        for key, value in snapshot.items():
            state[key] = value

    def clear_all_prompts_in_frame(self, inference_state: dict, frame_idx: int,
                                   obj_id: int, **_kwargs):
        inference_state["_prompt_records"].pop((int(frame_idx), int(obj_id)), None)

    def reset_state(self, inference_state: dict) -> None:
        preserved = {key: inference_state[key] for key in
                     ("images", "num_frames", "video_width", "video_height",
                      "_feature_cache")}
        inference_state.clear()
        inference_state.update({
            **preserved, "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(), "obj_ids": [],
            "point_inputs_per_obj": {}, "mask_inputs_per_obj": {},
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {}, "_tracker_state": None,
            "_prompt_records": OrderedDict(),
        })
        with trace_performance_call("sam31.clear_mlx_cache", device="cache", reason="reset"):
            self.mx.clear_cache()

    def release(self, inference_state: dict | None = None) -> None:
        if inference_state is not None:
            inference_state.get("_feature_cache", {}).clear()
            inference_state.clear()
        with trace_performance_call("sam31.clear_mlx_cache", device="cache", reason="release"):
            self.mx.clear_cache()


def build_mlx_sam31_predictor(**kwargs) -> Sam31MLXPredictor:
    return Sam31MLXPredictor(**kwargs)


def clear_mlx_sam31_cache() -> None:
    try:
        mx, *_ = _load_runtime()
        mx.clear_cache()
    except Exception:
        pass
