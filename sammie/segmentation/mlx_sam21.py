"""Memory-bounded SAM 2.1 video tracking on Apple Silicon through MLX."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import platform
import sys
from typing import Iterator

import numpy as np
from PIL import Image


MLX_MODEL_NAME = "Base+ MLX FP16"
MODEL_ID = "facebook/sam2.1-hiera-base-plus"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "third_party" / "sam2-mlx"
RUNTIME_SRC = RUNTIME_ROOT / "src"
OFFICIAL_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
MLX_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_base_plus_mlx_fp16.safetensors"
MIB = 1024 * 1024


def is_mlx_sam21_model(name: str | None) -> bool:
    return name == MLX_MODEL_NAME


def _load_runtime():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("SAM 2.1 MLX requires an Apple Silicon Mac.")
    if not (RUNTIME_SRC / "mlx_sam" / "video_predictor.py").exists():
        raise RuntimeError(
            "The pinned SAM 2 MLX runtime is missing. Run "
            "`git submodule update --init --recursive` from the Sammie folder."
        )
    runtime_src = str(RUNTIME_SRC)
    if runtime_src not in sys.path:
        sys.path.insert(0, runtime_src)

    import mlx.core as mx
    from mlx_sam.preprocess import preprocess_image
    from mlx_sam.video_predictor import SAM2VideoPredictor, _best_low_mask

    return mx, preprocess_image, SAM2VideoPredictor, _best_low_mask


class LazyFrameSequence:
    """Array-like, bounded LRU view over Sammie's extracted image sequence.

    The upstream MLX predictor only requires ``shape`` and slice indexing.
    Supplying those lazily avoids its default all-frames float32 allocation.
    """

    def __init__(self, frame_dir: str | Path, image_size: int, cache_frames: int = 3):
        self.frame_dir = Path(frame_dir)
        self.image_size = int(image_size)
        self.cache_frames = max(1, int(cache_frames))
        self.paths = sorted(
            path
            for path in self.frame_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise ValueError(f"No image frames found in directory: {self.frame_dir}")
        with Image.open(self.paths[0]) as first:
            self.video_width, self.video_height = first.size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._preprocess = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.paths), 3, self.image_size, self.image_size)

    @property
    def nbytes(self) -> int:
        """Resident bytes, rather than the virtual size of the whole clip."""
        return sum(frame.nbytes for frame in self._cache.values())

    def __len__(self) -> int:
        return len(self.paths)

    def _get(self, index: int) -> np.ndarray:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        if self._preprocess is None:
            _, self._preprocess, _, _ = _load_runtime()
        with Image.open(self.paths[index]) as image:
            # Normalize in float32, then retain the prepared frame in FP16 so
            # inputs match the FP16 model and each 1024 frame costs 6 MiB rather
            # than 12 MiB in unified memory.
            frame = self._preprocess(image, image_size=self.image_size).astype(
                np.float16, copy=False
            )
        self._cache[index] = frame
        while len(self._cache) > self.cache_frames:
            self._cache.popitem(last=False)
        return frame

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get(key)[0]
        if not isinstance(key, slice):
            raise TypeError(f"Unsupported frame index type: {type(key)!r}")
        start, stop, step = key.indices(len(self))
        indexes = range(start, stop, step)
        frames = [self._get(index) for index in indexes]
        if not frames:
            return np.empty((0, 3, self.image_size, self.image_size), dtype=np.float32)
        return np.concatenate(frames, axis=0)

    def clear(self) -> None:
        self._cache.clear()


def _prune_non_conditioning_state(state: dict, current_frame: int, reverse: bool, window: int) -> None:
    """Keep the exact memory/pointer horizon SAM 2.1 can consult.

    Conditioning frames are never evicted. Completed full-resolution masks are
    already written by Sammie, while old non-conditioning tensors otherwise grow
    linearly with clip length in the upstream predictor.
    """

    window = max(16, int(window))
    if reverse:
        keep_min, keep_max = current_frame, current_frame + window - 1
    else:
        keep_min, keep_max = current_frame - window + 1, current_frame
    for output_dict in state.get("output_dict_per_obj", {}).values():
        non_cond = output_dict.get("non_cond_frame_outputs", {})
        for frame_idx in list(non_cond):
            if not (keep_min <= frame_idx <= keep_max):
                non_cond.pop(frame_idx, None)


def _predict_initial_frame(model, encoded, point_inputs, prev_low, predict_initial, mx):
    """Decode an initial prompt or refine it using the prior mask as input."""

    if prev_low is None:
        return predict_initial(
            encoded,
            point_inputs["point_coords"],
            point_inputs["point_labels"],
        )
    return model.predict_from_encoded(
        encoded,
        mx.array(point_inputs["point_coords"][None].astype(np.float32)),
        mx.array(point_inputs["point_labels"][None].astype(np.int32)),
        mask_input=prev_low,
        multimask_output=False,
        add_no_mem_embed=True,
    )


def build_mlx_sam21_predictor(
    *,
    cache_limit_mib: int = 256,
    memory_limit_mib: int = 6144,
    frame_cache_size: int = 3,
    state_window: int = 16,
):
    """Create the optimized 1024-resolution Base+ FP16 predictor."""

    mx, _, SAM2VideoPredictor, best_low_mask = _load_runtime()
    from .model_conversion import ensure_fp16_checkpoint

    mx.set_cache_limit(max(64, int(cache_limit_mib)) * MIB)
    mx.set_memory_limit(max(2048, int(memory_limit_mib)) * MIB)
    checkpoint = ensure_fp16_checkpoint(OFFICIAL_CHECKPOINT, MLX_CHECKPOINT)

    class SammieMLXVideoPredictor(SAM2VideoPredictor):
        def _run_single_frame_inference(
            self,
            inference_state: dict,
            obj_idx: int,
            frame_idx: int,
            is_init_cond_frame: bool,
            point_inputs: dict | None,
            reverse: bool,
            prev_low=None,
            run_mem_encoder: bool = True,
            memory_frame_idx: int | None = None,
        ) -> dict:
            """Run one frame, including repeated-click refinement on frame zero.

            The pinned runtime tries to use temporal memory whenever ``prev_low``
            is present. During initial-frame refinement that memory has deliberately
            not been encoded yet, so its value is ``None``. Meta's predictor instead
            runs the image decoder again and supplies the previous low-resolution
            mask as a mask prompt. Keep that behavior in the Sammie adapter without
            modifying the pinned dependency.
            """

            encoded = self._get_image_feature(inference_state, frame_idx)
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if is_init_cond_frame and point_inputs is not None:
                out = _predict_initial_frame(
                    self.model,
                    encoded,
                    point_inputs,
                    prev_low,
                    self._predict_initial,
                    mx,
                )
            else:
                out = self._predict_tracked(
                    encoded,
                    obj_output_dict,
                    frame_idx,
                    point_inputs=point_inputs,
                    prev_low=prev_low,
                    reverse=reverse,
                    memory_frame_idx=memory_frame_idx,
                )

            low = best_low_mask(out)
            current_out = {
                "pred_masks": low,
                "obj_ptr": out["obj_ptr"],
                "object_score_logits": out["object_score_logits"],
                "maskmem_features": None,
                "maskmem_pos_enc": None,
            }
            if run_mem_encoder:
                memory = self._encode_memory(
                    encoded,
                    low,
                    out,
                    frame_idx if memory_frame_idx is None else memory_frame_idx,
                    is_mask_from_points=point_inputs is not None,
                )
                current_out.update(memory)
            eval_targets = [
                current_out["pred_masks"],
                current_out["obj_ptr"],
                current_out["object_score_logits"],
            ]
            if current_out["maskmem_features"] is not None:
                eval_targets.extend(
                    [current_out["maskmem_features"], current_out["maskmem_pos_enc"]]
                )
            mx.eval(*eval_targets)
            return current_out

        def init_state(
            self,
            video_path,
            offload_video_to_cpu: bool = False,
            offload_state_to_cpu: bool = False,
            async_loading_frames: bool = False,
            precompute_image_features: bool = False,
            feature_batch_size: int = 1,
        ) -> dict:
            if precompute_image_features:
                raise ValueError(
                    "Full-video feature precomputation is disabled in Sammie's "
                    "memory-bounded MLX backend."
                )
            frames = LazyFrameSequence(
                video_path, image_size=self.image_size, cache_frames=frame_cache_size
            )
            state = {
                "images": frames,
                "num_frames": len(frames),
                "offload_video_to_cpu": False,
                "offload_state_to_cpu": False,
                "async_loading_frames": False,
                "keep_image_features": False,
                "video_height": frames.video_height,
                "video_width": frames.video_width,
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": OrderedDict(),
                "obj_idx_to_id": OrderedDict(),
                "obj_ids": [],
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
            }
            # Warm the graph and retain only frame zero's current feature.
            self._get_image_feature(state, 0)
            mx.synchronize()
            return state

        def propagate_in_video(self, inference_state: dict, *args, **kwargs) -> Iterator:
            reverse = bool(kwargs.get("reverse", False))
            for frame_idx, obj_ids, masks in super().propagate_in_video(
                inference_state, *args, **kwargs
            ):
                yield frame_idx, obj_ids, masks
                _prune_non_conditioning_state(
                    inference_state, int(frame_idx), reverse, state_window
                )
                # MLX keeps reusable allocations under cache_limit; periodically
                # release anything above that bound after results reach the caller.
                if int(frame_idx) % 30 == 0:
                    mx.clear_cache()

        def release(self, inference_state: dict | None = None) -> None:
            if inference_state is not None:
                images = inference_state.get("images")
                if hasattr(images, "clear"):
                    images.clear()
                inference_state.clear()
            mx.clear_cache()

    predictor = SammieMLXVideoPredictor(
        checkpoint=checkpoint,
        model_id=MODEL_ID,
        image_size=1024,
        memory_dtype="float16",
        memory_attention_dtype="float16",
        num_maskmem=7,
        max_obj_ptrs_in_encoder=16,
        add_all_frames_to_correct_as_cond=True,
    )
    mx.eval(predictor.model.parameters())
    return predictor


def clear_mlx_cache() -> None:
    try:
        mx, _, _, _ = _load_runtime()
        mx.clear_cache()
    except Exception:
        pass
