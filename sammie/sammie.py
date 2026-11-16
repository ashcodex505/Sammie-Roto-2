# sammie/sammie.py
import cv2
import os
import numpy as np
import shutil
import re
import glob
import zipfile
import threading
import queue
import gc
from functools import wraps
import av
from tqdm import tqdm
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from sam2.build_sam import build_sam2_video_predictor
from sammie import core
from sammie.smooth import run_smoothing_model, prepare_smoothing_model
from sammie.duplicate_frame_handler import replace_similar_matte_frames
from sammie.settings_manager import get_settings_manager
from sammie.gui_widgets import show_message_dialog
from sammie.model_downloader import ensure_models
from sammie.segmentation import (
    MLX_MODEL_NAME,
    build_mlx_sam21_predictor,
    is_mlx_sam21_model,
)
from sammie.segmentation.mlx_sam21 import clear_mlx_cache

smoothing_model = None  # global variable needed to avoid complexity of passing the model around


def _segmentation_inference_context(function):
    """Use Torch inference mode only for Torch-backed segmentation models."""

    @wraps(function)
    def wrapped(manager, *args, **kwargs):
        if is_mlx_sam21_model(manager.loaded_model_name):
            return function(manager, *args, **kwargs)
        with core.DeviceManager.inference_context():
            return function(manager, *args, **kwargs)

    return wrapped


def _binary_mask_numpy(mask_logits) -> np.ndarray:
    """Convert either Torch tensors or NumPy/MLX results into an 8-bit mask."""

    binary = mask_logits > 0.0
    if hasattr(binary, "detach"):
        binary = binary.detach()
    if hasattr(binary, "cpu"):
        binary = binary.cpu()
    if hasattr(binary, "numpy"):
        binary = binary.numpy()
    return np.asarray(binary).squeeze().astype(np.uint8) * 255


def _mask_motion_envelope(mask: np.ndarray, motion_radius: int) -> np.ndarray:
    """Return a bounded-resolution dilation around a full-resolution mask."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    radius = max(0, int(motion_radius))
    if radius == 0 or not np.any(binary):
        return binary.astype(bool)

    height, width = binary.shape
    scale = max(1, int(np.ceil(max(height, width) / 960.0)))
    small_width = max(1, int(np.ceil(width / scale)))
    small_height = max(1, int(np.ceil(height / scale)))
    small_mask = cv2.resize(
        binary, (small_width, small_height), interpolation=cv2.INTER_AREA
    ) > 0
    small_radius = max(1, int(np.ceil(radius / scale)))
    kernel_size = small_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    small_envelope = cv2.dilate(
        small_mask.astype(np.uint8), kernel, iterations=1
    )
    return cv2.resize(
        small_envelope, (width, height), interpolation=cv2.INTER_NEAREST
    ) > 0


def _released_foreground_region(selected_mask: np.ndarray,
                                previous_foreground: np.ndarray,
                                current_foreground: np.ndarray,
                                motion_radius: int) -> np.ndarray:
    """Recover a stable region where a separately tracked occluder moved away."""
    previous = np.asarray(previous_foreground) > 0
    current = np.asarray(current_foreground) > 0
    released = previous & ~current
    nearby = _mask_motion_envelope(selected_mask, motion_radius)
    return (released & nearby).astype(np.uint8) * 255


def _stabilize_propagated_mask(predicted_mask: np.ndarray,
                               previous_mask: np.ndarray,
                               motion_radius: int) -> np.ndarray:
    """Trim sudden growth outside the selected region's motion envelope.

    If overlap is too small, the guard returns the model prediction intact. That
    fail-open behavior lets cuts and unusually large motion recover instead of
    erasing the object completely.
    """
    predicted = (np.asarray(predicted_mask) > 0).astype(np.uint8)
    previous = (np.asarray(previous_mask) > 0).astype(np.uint8)
    if not np.any(predicted) or not np.any(previous):
        return predicted * 255

    envelope = _mask_motion_envelope(previous, motion_radius)
    stabilized = predicted & envelope

    # A poor overlap signals a cut or fast motion.  Keep the model result so the
    # tracker can recover rather than replacing the object with an empty mask.
    kept_fraction = np.count_nonzero(stabilized) / max(
        1, np.count_nonzero(predicted)
    )
    if kept_fraction < 0.35:
        return predicted * 255

    return stabilized.astype(np.uint8) * 255

# .........................................................................................
# SAM2 segmentation
# .........................................................................................

class SamManager(core.CallbackMixin):
    def __init__(self):
        super().__init__()
        self.model = None
        self.loaded_model_name = None
        self.predictor = None
        self.inference_state = None
        self.propagated = False  # whether we have propagated the masks
        self.deduplicated = False  # whether we have deduplicated the masks

    def load_segmentation_model(self, model=None, parent_window=None):
        if model is None:
            settings_mgr = get_settings_manager()
            sam_model = settings_mgr.get_session_setting("sam_model", "Base")
        else:
            sam_model = model
        # Migrate sessions saved before retired model choices were removed.
        if sam_model == "Efficient":
            print("EfficientTAM is no longer available; using SAM2 Base instead")
            sam_model = "Base"
        elif sam_model == "Large":
            print(f"{sam_model} is no longer available; using SAM 2.1 Base+ MLX FP16")
            sam_model = MLX_MODEL_NAME
        core.DeviceManager.clear_cache()
        if is_mlx_sam21_model(sam_model):
            if not ensure_models("Base", parent=parent_window):
                return False
            settings_mgr = get_settings_manager()
            self.predictor = build_mlx_sam21_predictor(
                cache_limit_mib=settings_mgr.get_app_setting("mlx_cache_limit_mib", 256),
                memory_limit_mib=settings_mgr.get_app_setting("mlx_memory_limit_mib", 6144),
                frame_cache_size=settings_mgr.get_app_setting("mlx_frame_cache_size", 3),
                state_window=settings_mgr.get_app_setting("mlx_state_window", 16),
            )
            self.loaded_model_name = sam_model
            print("Loaded SAM 2.1 Base+ MLX FP16 model (1024px)")
            return True

        device = core.DeviceManager.get_device()
        if sam_model == "Base":
            print("Loaded SAM2 Base model")
            checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif sam_model == "Anime":
            print("Loaded SAM2 Anime model")
            checkpoint = "./checkpoints/sam2.1_anime_v1.pt"
            model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"

        # Check if files exist
        if not ensure_models(sam_model, parent=parent_window):
            return False

        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        self.loaded_model_name = sam_model
        return True  # model loaded successfully

    def unload_segmentation_model(self):
        """Unload the SAM model and clear cache"""
        was_mlx = is_mlx_sam21_model(self.loaded_model_name)
        if was_mlx and self.predictor is not None:
            release = getattr(self.predictor, "release", None)
            if release is not None:
                release(self.inference_state)
        self.predictor = None
        self.inference_state = None
        self.loaded_model_name = None
        gc.collect()
        if was_mlx:
            clear_mlx_cache()
        core.DeviceManager.clear_cache()
        print("Unloaded Segmentation model")

    def offload_model_to_cpu(self):
        """Offload SAM2 model to CPU to free VRAM"""
        if is_mlx_sam21_model(self.loaded_model_name):
            clear_mlx_cache()
            return
        device = core.DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do

        if self.predictor is not None:
            self.predictor.to('cpu')
            core.DeviceManager.clear_cache()

    def load_model_to_device(self):
        """Load SAM2 model back to the active device"""
        if is_mlx_sam21_model(self.loaded_model_name):
            return
        device = core.DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do

        if self.predictor is not None:
            self.predictor.to(device)

    def initialize_predictor(self):
        self.inference_state = self.predictor.init_state(
            video_path=core.frames_dir, async_loading_frames=True, offload_video_to_cpu=True
        )

    def _clear_frame_if_tracked(self, object_id, frame_number):
        """Reset frame_number to a blank slate if it is been tracked,
        so new points generate a segment without using any memory from other frames
        """
        clear_hook = getattr(self.predictor, "clear_frame_if_tracked", None)
        if clear_hook is not None and clear_hook(
                self.inference_state, object_id, frame_number):
            return
        obj_idx = self.inference_state["obj_id_to_idx"].get(object_id)
        if obj_idx is None:
            return
        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        if frame_number not in tracked:
            return
        self.predictor.clear_all_prompts_in_frame(
            self.inference_state, frame_number, object_id, need_output=False
        )
        tracked.pop(frame_number, None)
        for d in (self.inference_state["output_dict_per_obj"][obj_idx],
                  self.inference_state["temp_output_dict_per_obj"][obj_idx]):
            d["non_cond_frame_outputs"].pop(frame_number, None)

    def _snapshot_frame_state(self, object_id, frame_number):
        """Capture everything _clear_frame_if_tracked (and add_new_points_or_box) could
        touch for this frame/object, so a preview can be undone without a trace.
        """
        snapshot_hook = getattr(self.predictor, "snapshot_frame_state", None)
        if snapshot_hook is not None:
            return {"_predictor_snapshot": snapshot_hook(
                self.inference_state, object_id, frame_number)}
        obj_idx = self.inference_state["obj_id_to_idx"].get(object_id)
        if obj_idx is None:
            # Object doesn't exist yet - add_new_points_or_box() will auto-register it
            # as a side effect of running the preview (obj_id_to_idx/obj_idx_to_id/
            # obj_ids, plus empty entries in every per-object dict below). Remember
            # that so _restore_frame_state can undo the registration afterward,
            # instead of leaving a brand-new, populated object behind permanently.
            return {"object_id": object_id, "obj_idx": None, "was_registered": False}

        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        temp_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        point_inputs = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs = self.inference_state["mask_inputs_per_obj"][obj_idx]
        return {
            "object_id": object_id,
            "obj_idx": obj_idx,
            "was_registered": True,
            "was_tracked": frame_number in tracked,
            "tracked_entry": tracked.get(frame_number),
            "output_cond": output_dict["cond_frame_outputs"].get(frame_number),
            "output_noncond": output_dict["non_cond_frame_outputs"].get(frame_number),
            "temp_cond": temp_dict["cond_frame_outputs"].get(frame_number),
            "temp_noncond": temp_dict["non_cond_frame_outputs"].get(frame_number),
            "had_point_inputs": frame_number in point_inputs,
            "point_inputs": point_inputs.get(frame_number),
            "had_mask_inputs": frame_number in mask_inputs,
            "mask_inputs": mask_inputs.get(frame_number),
        }

    def _restore_frame_state(self, frame_number, snapshot):
        """Undo _snapshot_frame_state - restores frames_tracked_per_obj plus every
        cond/non-cond entry exactly as it was, regardless of what happened in between."""
        if snapshot is None:
            return

        if "_predictor_snapshot" in snapshot:
            self.predictor.restore_frame_state(
                self.inference_state, snapshot["_predictor_snapshot"])
            return

        object_id = snapshot["object_id"]

        if not snapshot["was_registered"]:
            # The object didn't exist before the preview ran - undo whatever
            # auto-registration add_new_points_or_box() performed as a side effect,
            # rather than leaving a phantom object (with real prompt/output data for
            # this frame) sitting in inference_state for the rest of the session.
            obj_idx = self.inference_state["obj_id_to_idx"].pop(object_id, None)
            if obj_idx is not None:
                self.inference_state["obj_idx_to_id"].pop(obj_idx, None)
                obj_ids = self.inference_state.get("obj_ids")
                if obj_ids is not None and object_id in obj_ids:
                    obj_ids.remove(object_id)
                for dict_name in (
                    "output_dict_per_obj", "temp_output_dict_per_obj",
                    "frames_tracked_per_obj", "point_inputs_per_obj", "mask_inputs_per_obj",
                ):
                    d = self.inference_state.get(dict_name)
                    if d is not None:
                        d.pop(obj_idx, None)
            return

        obj_idx = snapshot["obj_idx"]
        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        temp_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        point_inputs = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if snapshot["was_tracked"]:
            tracked[frame_number] = snapshot["tracked_entry"]
        else:
            tracked.pop(frame_number, None)

        for d, cond_key, noncond_key in (
            (output_dict, "output_cond", "output_noncond"),
            (temp_dict, "temp_cond", "temp_noncond"),
        ):
            for storage_key, snap_key in (("cond_frame_outputs", cond_key), ("non_cond_frame_outputs", noncond_key)):
                val = snapshot[snap_key]
                if val is not None:
                    d[storage_key][frame_number] = val
                else:
                    d[storage_key].pop(frame_number, None)

        if snapshot["had_point_inputs"]:
            point_inputs[frame_number] = snapshot["point_inputs"]
        else:
            point_inputs.pop(frame_number, None)

        if snapshot["had_mask_inputs"]:
            mask_inputs[frame_number] = snapshot["mask_inputs"]
        else:
            mask_inputs.pop(frame_number, None)

    @_segmentation_inference_context
    def segment_image(self, frame_number, object_id, input_points, input_labels):
        extension = core.get_frame_extension()
        frame_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
        if os.path.exists(frame_filename):
            self._clear_frame_if_tracked(object_id, frame_number)

            # Run segmentation function
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=input_points,
                labels=input_labels,
                clear_old_points=True,
            )
            # Save the segmentation mask for the object we actually edited.
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id != object_id:
                    continue
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                mask = _binary_mask_numpy(out_mask_logits[i])
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)

            # Notify that segmentation is complete
            self._notify('segmentation_complete', frame=frame_number, object_id=object_id, out_obj_ids=out_obj_ids)

    @_segmentation_inference_context
    def preview_point(self, frame_number, object_id, all_points, preview_x, preview_y, is_positive):
        """Run a preview using the video predictor, then revert the state."""
        if self.predictor is None or self.inference_state is None:
            return None

        snapshot = self._snapshot_frame_state(object_id, frame_number)

        try:
            self._clear_frame_if_tracked(object_id, frame_number)

            existing = [p for p in all_points
                        if p['frame'] == frame_number and p['object_id'] == object_id]

            # Build preview point set
            preview_points = np.array([[p['x'], p['y']] for p in existing] + [[preview_x, preview_y]], dtype=np.float32)
            preview_labels = np.array([1 if p['positive'] else 0 for p in existing] + [1 if is_positive else 0], dtype=np.int32)

            # run with preview point
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=preview_points,
                labels=preview_labels,
                clear_old_points=True,
            )

            # Capture the preview mask
            preview_mask = None
            for i, oid in enumerate(out_obj_ids):
                if oid == object_id:
                    preview_mask = _binary_mask_numpy(out_mask_logits[i])
                    break

            return preview_mask

        except Exception as e:
            print(f"Preview error: {e}")
            return None

        finally:
            self._restore_frame_state(frame_number, snapshot)

    @_segmentation_inference_context
    def replay_points(self, points_list):
        """Rebuild prompt state with one model call per object and frame.

        ``add_new_points_or_box(clear_old_points=True)`` accepts the complete
        prompt set, which is also what ``segment_image`` submits after a click.
        Replaying every cumulative prefix made an undo with N points perform
        1+2+...+N prompt work and needlessly blocked the UI.
        """
        self.predictor.reset_state(self.inference_state)

        grouped_points = {}
        for point in points_list:
            key = (point['frame'], point['object_id'])
            grouped_points.setdefault(key, []).append(
                (point['x'], point['y'], point['positive'])
            )

        for (frame_number, object_id), filtered_points in sorted(grouped_points.items()):
            input_points = np.array(
                [(x, y) for x, y, _ in filtered_points], dtype=np.float32
            )
            input_labels = np.array(
                [1 if positive else 0 for _, _, positive in filtered_points],
                dtype=np.int32,
            )
            try:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_number,
                    obj_id=object_id,
                    points=input_points,
                    labels=input_labels,
                    clear_old_points=True,
                )
            except Exception as error:
                print(
                    f"Error replaying {len(filtered_points)} points for frame "
                    f"{frame_number}, object {object_id}: {error}"
                )
                continue

            # Save only the mask for the object whose prompts were replayed.
            for j, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id != object_id:
                    continue
                mask_filename = os.path.join(
                    core.mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png"
                )
                mask = _binary_mask_numpy(out_mask_logits[j])
                try:
                    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                    cv2.imwrite(mask_filename, mask)
                except Exception as error:
                    print(
                        f"Error saving mask for frame {frame_number}, "
                        f"object {out_obj_id}: {error}"
                    )

        self._notify('replay_complete')

    @_segmentation_inference_context
    def replace_frame_object_points(self, frame_number, object_id, points_list):
        """Replace one frame/object prompt group without replaying the project."""
        input_points = np.asarray(
            [(point['x'], point['y']) for point in points_list],
            dtype=np.float32,
        ).reshape(-1, 2)
        input_labels = np.asarray(
            [1 if point['positive'] else 0 for point in points_list],
            dtype=np.int32,
        )

        replace_hook = getattr(self.predictor, "replace_points_in_frame", None)
        if replace_hook is not None:
            result = replace_hook(
                self.inference_state,
                frame_number,
                object_id,
                input_points,
                input_labels,
            )
        else:
            result = self._replace_generic_frame_object_points(
                frame_number, object_id, input_points, input_labels
            )

        out_obj_ids = []
        if result is not None:
            _, out_obj_ids, out_mask_logits = result
            if len(input_labels):
                for index, out_obj_id in enumerate(out_obj_ids):
                    if out_obj_id != object_id:
                        continue
                    mask_filename = os.path.join(
                        core.mask_dir,
                        f"{frame_number:05d}",
                        f"{out_obj_id}.png",
                    )
                    mask = _binary_mask_numpy(out_mask_logits[index])
                    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                    cv2.imwrite(mask_filename, mask)

        self._notify(
            'segmentation_complete',
            frame=frame_number,
            object_id=object_id,
            out_obj_ids=out_obj_ids,
        )

    def _replace_generic_frame_object_points(
            self, frame_number, object_id, input_points, input_labels):
        """Targeted prompt replacement for SAM 2/2.1 predictor state."""
        state = self.inference_state
        obj_idx = state.get("obj_id_to_idx", {}).get(object_id)
        if obj_idx is not None:
            self.predictor.clear_all_prompts_in_frame(
                state, frame_number, object_id, need_output=False
            )
            for collection_name in (
                    "output_dict_per_obj", "temp_output_dict_per_obj"):
                output_dict = state.get(collection_name, {}).get(obj_idx, {})
                output_dict.get("cond_frame_outputs", {}).pop(frame_number, None)
                output_dict.get("non_cond_frame_outputs", {}).pop(frame_number, None)
            state.get("frames_tracked_per_obj", {}).get(obj_idx, {}).pop(
                frame_number, None
            )

        if not len(input_labels):
            return None
        return self.predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_number,
            obj_id=object_id,
            points=input_points,
            labels=input_labels,
            clear_old_points=True,
        )


    @_segmentation_inference_context
    def _propagate(self, parent_window, start_frame_idx, max_frame_num_to_track, reverse=False,
                    show_progress=True):
        """Core propagation loop shared by all tracking functions.

        Args:
            parent_window: Used for the progress dialog and to nudge the frame slider as we go.
            start_frame_idx: Frame to start propagating from.
            max_frame_num_to_track: How many additional frames to propagate beyond the start
                frame, or None to propagate to the end (or beginning, if reverse=True) of the video.
            reverse: If True, propagate backward toward frame 0 instead of forward.
            show_progress: If False, skips the progress dialog - intended for single-frame steps
                where a modal dialog would just be visual noise.

        Returns:
            (last_frame_idx, cancelled) - last_frame_idx is the last frame actually processed
            (None if nothing was processed), cancelled is True if the user hit Cancel.
        """
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        total_frames = (max_frame_num_to_track + 1) if max_frame_num_to_track is not None else core.VideoInfo.total_frames

        progress_dialog = None
        if show_progress:
            progress_dialog = QProgressDialog("Tracking...", "Cancel", 0, 100, parent_window)
            progress_dialog.setWindowTitle("Progress")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setAutoClose(True)
            progress_dialog.show()

        last_frame_idx = None
        cancelled = False
        selected_object_id = settings_mgr.get_session_setting(
            "selected_object_id", None
        )
        selected_only = settings_mgr.get_session_setting(
            "track_selected_object_only", True
        )
        prevent_drift = settings_mgr.get_session_setting(
            "prevent_mask_drift", True
        )
        previous_masks = {}
        video_size = max(
            int(getattr(core.VideoInfo, "width", 0) or 0),
            int(getattr(core.VideoInfo, "height", 0) or 0),
            1,
        )
        motion_radius = max(4, int(round(video_size * 0.02)))
        previous_foreground = None

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track, reverse=reverse):
            for i, out_obj_id in enumerate(out_obj_ids):
                if (selected_only and selected_object_id is not None
                        and out_obj_id != selected_object_id):
                    continue
                mask_filename = os.path.join(core.mask_dir, f"{out_frame_idx:05d}", f"{out_obj_id}.png")
                mask = _binary_mask_numpy(out_mask_logits[i])

                obj_idx = self.inference_state.get(
                    "obj_id_to_idx", {}
                ).get(out_obj_id)
                point_inputs = self.inference_state.get(
                    "point_inputs_per_obj", {}
                ).get(obj_idx, {}) if obj_idx is not None else {}
                is_prompt_anchor = out_frame_idx in point_inputs
                previous = previous_masks.get(out_obj_id)
                if (prevent_drift and out_obj_id == selected_object_id
                        and previous is not None and not is_prompt_anchor):
                    mask = _stabilize_propagated_mask(
                        mask, previous, motion_radius
                    )

                if (prevent_drift and out_obj_id == selected_object_id):
                    # Preserve an opening/background region while respecting
                    # separately tracked foreground objects.  This makes a
                    # locked window mask follow around people instead of
                    # painting over them as the prior shape is retained.
                    frame_mask_dir = os.path.dirname(mask_filename)
                    current_foreground = np.zeros_like(mask, dtype=np.uint8)
                    for other_filename in glob.glob(
                            os.path.join(frame_mask_dir, "*.png")):
                        try:
                            other_id = int(os.path.splitext(
                                os.path.basename(other_filename)
                            )[0])
                        except ValueError:
                            continue
                        if other_id == out_obj_id:
                            continue
                        other_mask = cv2.imread(
                            other_filename, cv2.IMREAD_GRAYSCALE
                        )
                        if other_mask is not None and other_mask.shape == mask.shape:
                            current_foreground[other_mask > 0] = 255
                    if previous is not None and previous_foreground is not None:
                        mask |= _released_foreground_region(
                            previous,
                            previous_foreground,
                            current_foreground,
                            motion_radius,
                        )
                    mask[current_foreground > 0] = 0
                    previous_foreground = current_foreground
                previous_masks[out_obj_id] = mask
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)

            last_frame_idx = out_frame_idx

            if progress_dialog is not None:
                frames_processed = abs(out_frame_idx - start_frame_idx) + 1
                progress_dialog.setValue(int(frames_processed * 100 / total_frames))

            # Update display at the specified frequency (always update for quick, dialog-less steps)
            if not show_progress or out_frame_idx % display_update_frequency == 0:
                try:
                    parent_window.frame_slider.setValue(out_frame_idx)
                except Exception as e:
                    print(f"Error updating display: {e}")

            QApplication.processEvents()
            if progress_dialog is not None and progress_dialog.wasCanceled():
                cancelled = True
                break

        if progress_dialog is not None:
            if cancelled:
                progress_dialog.close()
            else:
                progress_dialog.setValue(100)

        # MLX can retain temporary graph allocations from the last inference
        # frame until the next periodic cache sweep. Release those immediately
        # when a tracking run ends so memory pressure drops while the user
        # reviews or edits the result. Temporal state needed for corrections is
        # kept in inference_state and is not affected by clearing this cache.
        if is_mlx_sam21_model(self.loaded_model_name):
            clear_mlx_cache()

        # Always leave the slider showing the last frame actually processed
        if last_frame_idx is not None:
            try:
                parent_window.frame_slider.setValue(last_frame_idx)
                QApplication.processEvents()
            except Exception as e:
                print(f"Error updating display: {e}")

        return last_frame_idx, cancelled

    def track_objects(self, parent_window):
        """Track all objects across the full in/out point range (or the entire video)."""
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        if in_point is None:
            in_point = 0
        frames_to_track = None
        total_frames = frame_count
        if out_point is not None:
            frames_to_track = out_point - in_point
            total_frames = frames_to_track + 1

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=in_point, max_frame_num_to_track=frames_to_track, reverse=False)

        if not cancelled:
            self.propagated = (total_frames == frame_count)
            print("Tracking completed")
            return 1
        else:
            self.propagated = False
            print("Tracking cancelled")
            return 0

    def track_forward(self, parent_window, current_frame):
        """Track all objects forward from current_frame to the out point (or end of video)."""
        settings_mgr = get_settings_manager()
        out_point = settings_mgr.get_session_setting("out_point", None)
        last_frame = out_point if out_point is not None else core.VideoInfo.total_frames - 1
        if current_frame >= last_frame:
            print("Already at the last frame")
            return 1

        max_frame_num_to_track = max(last_frame - current_frame, 0)

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=max_frame_num_to_track,
            reverse=False)

        if cancelled:
            print("Forward tracking cancelled")
            return 0
        print(f"Forward tracking completed up to frame {last_frame_idx}")
        return 1

    def track_backward(self, parent_window, current_frame):
        """Track all objects backward from current_frame to the in point (or start of video)."""
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        if in_point is None:
            in_point = 0
        if current_frame <= in_point:
            print("Already at the first frame")
            return 1

        max_frame_num_to_track = max(current_frame - in_point, 0)

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=max_frame_num_to_track,
            reverse=True)

        if cancelled:
            print("Backward tracking cancelled")
            return 0
        print(f"Backward tracking completed back to frame {last_frame_idx}")
        return 1

    def track_one_frame_forward(self, parent_window, current_frame):
        """Track all objects one frame forward from current_frame. Returns the new frame index."""
        last_frame = core.VideoInfo.total_frames - 1
        if current_frame >= last_frame:
            print("Already at the last frame")
            return current_frame

        last_frame_idx, _ = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=1,
            reverse=False, show_progress=False)

        return last_frame_idx if last_frame_idx is not None else current_frame

    def track_one_frame_backward(self, parent_window, current_frame):
        """Track all objects one frame backward from current_frame. Returns the new frame index."""
        if current_frame <= 0:
            print("Already at the first frame")
            return current_frame

        last_frame_idx, _ = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=1,
            reverse=True, show_progress=False)

        return last_frame_idx if last_frame_idx is not None else current_frame

    def clear_tracking(self):
        """Clear tracking data by deleting all masks, this needs to be followed up by replay_points"""
        if os.path.exists(core.mask_dir):
            shutil.rmtree(core.mask_dir)
        os.makedirs(core.mask_dir)
        self.predictor.reset_state(self.inference_state)
        core.DeviceManager.clear_cache()
        if self.propagated:
            print("Tracking data cleared")
        self.propagated = False
        self.deduplicated = False


# .........................................................................................
# Smoothing model
# .........................................................................................

def load_smoothing_model():
    global smoothing_model
    if smoothing_model is None:
        device = core.DeviceManager.get_device()
        try:
            smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)
        except Exception as e:
            print(f"Warning: Could not load antialiasing model: {e}")
            smoothing_model = None


# .........................................................................................
# View / display handlers
# .........................................................................................

def update_image(slider_value, view_options, points, return_numpy=False, object_id_filter=None, preview_mask=None, preview_object_id=None):
    """Main image update function - delegates to specific view handlers

    Args:
        slider_value: Frame number
        view_options: Dictionary of view options
        points: List of point dictionaries
        return_numpy: If True, return numpy array; if False, return QPixmap
        object_id_filter: If specified, only process masks for this object ID
        preview_mask: If specified, a live (uncommitted) mask to substitute
            for preview_object_id's normal saved mask
        preview_object_id: Which object preview_mask belongs to - every
            other object's normal saved mask is still loaded and shown as
            usual, unaffected by the preview

    Returns:
        QPixmap or numpy array depending on return_numpy parameter
    """
    view_mode = view_options.get("view_mode", "Segmentation-Edit")

    if view_mode == "Segmentation-Edit":
        return _handle_segmentation_edit_view(slider_value, view_options, points, return_numpy, object_id_filter, preview_mask, preview_object_id)
    elif view_mode == "Segmentation-Matte":
        return _handle_segmentation_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-BGcolor":
        return _handle_segmentation_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-Alpha":
        return _handle_segmentation_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Matte":
        return _handle_matting_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-BGcolor":
        return _handle_matting_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Alpha":
        return _handle_matting_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "ObjectRemoval":
        return _handle_object_removal_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "None":
        return _handle_none_view(slider_value, return_numpy)
    else:
        print(f"Unknown view mode: {view_mode}")
        return None


def load_removal_frame(frame_number):
    """
    Load the object removal frame image from disk.
    If the frame does not exist, load the base frame instead.
    """
    frame_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return core.load_base_frame(frame_number)


def _convert_to_qpixmap(image):
    """Convert NumPy array to QPixmap"""
    if image is None:
        return QPixmap.fromImage(QImage())
    image = image.copy()
    height, width = image.shape[:2]

    if len(image.shape) == 2:  # Grayscale
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif image.shape[2] == 3:  # RGB
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:  # RGBA
        bytes_per_line = 4 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

    return QPixmap.fromImage(q_image)


def _get_display_mask(frame_number, points, object_id_filter, folder, postprocess_fn):
    """
    Load the combined mask for a view and postprocess it. Returns None (without
    calling postprocess_fn) if no mask exists for this frame/object filter —
    callers are responsible for deciding what to do in that case.
    """
    mask = core.load_masks_for_frame(frame_number, points, return_combined=True,
                                      object_id_filter=object_id_filter, folder=folder)
    if mask is None:
        return None
    return postprocess_fn(mask)


def _maybe_antialias_mask(mask_3channel, view_options):
    """
    Apply the smoothing model to a 3-channel mask if antialiasing is enabled
    in view_options (default True) and the model is available. Used only by
    segmentation views — matting views have never applied antialiasing.
    """
    if not view_options.get("antialias", True):
        return mask_3channel
    global smoothing_model
    if smoothing_model is None:
        load_smoothing_model()
    if smoothing_model is not None:
        device = core.DeviceManager.get_device()
        mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)
    return mask_3channel


def _composite_bgcolor(image, mask, bgcolor):
    """Blend `image` with a solid `bgcolor` background using `mask` (single-channel, 0-255) as alpha."""
    bg = np.full_like(image, bgcolor)
    alpha = mask.astype(np.float32) / 255.0
    return cv2.blendLinear(image, bg, alpha, 1.0 - alpha)


def _composite_alpha(image, mask):
    """Merge `image` (RGB) and `mask` (single-channel, 0-255) into an RGBA image."""
    return cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2], mask])


def _handle_none_view(frame_number, return_numpy=False):
    """Handle None view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_edit_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None, preview_mask=None, preview_object_id=None):
    """Handle Segmentation-Edit view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    image = apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter, preview_mask, preview_object_id)

    highlighted_points = view_options.get('highlighted_point', None)

    if highlighted_points is None:
        highlighted_points = None
    elif isinstance(highlighted_points, list):
        highlighted_points = highlighted_points.copy()
    else:
        highlighted_points = [highlighted_points]

    image = draw_points(image, frame_number, points, highlighted_points)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Matte view"""
    mask = _get_display_mask(frame_number, points, object_id_filter, core.mask_dir, core.apply_mask_postprocessing)
    if mask is None:
        return None

    mask_3channel = np.stack([mask] * 3, axis=-1)
    mask_3channel = _maybe_antialias_mask(mask_3channel, view_options)

    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)


def _handle_segmentation_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-BGcolor view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.mask_dir, core.apply_mask_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    mask_3channel = np.stack([mask] * 3, axis=-1)
    mask_3channel = _maybe_antialias_mask(mask_3channel, view_options)

    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    image = _composite_bgcolor(image, mask_3channel[:, :, 0], bgcolor)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Alpha view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.mask_dir, core.apply_mask_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    mask_3channel = np.stack([mask] * 3, axis=-1)
    mask_3channel = _maybe_antialias_mask(mask_3channel, view_options)
    mask = mask_3channel[:, :, 0]

    image_rgba = _composite_alpha(image, mask)

    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)


def _handle_matting_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Matte view"""
    mask = _get_display_mask(frame_number, points, object_id_filter, core.matting_dir, core.apply_matany_postprocessing)
    if mask is None:
        return None

    mask_3channel = np.stack([mask] * 3, axis=-1)

    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)


def _handle_matting_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-BGcolor view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.matting_dir, core.apply_matany_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    image = _composite_bgcolor(image, mask, bgcolor)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_matting_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Alpha view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.matting_dir, core.apply_matany_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    image_rgba = _composite_alpha(image, mask)

    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)


def _handle_object_removal_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Object Removal view"""
    image = load_removal_frame(frame_number)
    if image is None:
        return None

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return None

    mask = core.apply_mask_postprocessing(mask)

    settings_mgr = get_settings_manager()
    grow = settings_mgr.get_session_setting("inpaint_grow", 0)
    mask = core.grow_shrink(mask, grow)

    if view_options.get("show_removal_mask", True):
        image = draw_removal_overlay(image, mask)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def draw_masks(image, processed_masks):
    """Draw masks on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    combined_colored_mask = np.zeros_like(image, dtype=np.uint8)
    mask_binary = np.zeros(image.shape[:2], dtype=bool)

    for object_id, mask in processed_masks.items():
        color = np.array(core.PALETTE[object_id % len(core.PALETTE)], dtype=np.uint8)
        mask_bin = mask > 0
        mask_binary |= mask_bin
        combined_colored_mask[mask_bin] = color

    if np.any(mask_binary):
        overlay = image.copy()
        overlay[mask_binary] = cv2.addWeighted(
            image[mask_binary], 0.5,
            combined_colored_mask[mask_binary], 0.5, 0
        )
        return overlay
    else:
        return image
    

def draw_removal_overlay(image, mask):
    """Draw an outline (no fill) around the removal mask, using object 0's color"""
    return draw_contours(image, {0: mask})


def draw_contours(image, processed_masks):
    """Draw colored contours on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    overlay = image.copy()
    kernel = np.ones((3, 3), np.uint8)

    for object_id, mask in processed_masks.items():
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        border_color = core.PALETTE[object_id % len(core.PALETTE)]
        overlay[edges > 0] = border_color

    return overlay


def draw_points(image, frame_number, points, highlighted_points=None):
    """Draw points on image"""
    frame_points = [p for p in points if p['frame'] == frame_number]
    if not frame_points:
        return image

    highlighted_set = set()
    if highlighted_points:
        highlighted_set = {(p['frame'], p['x'], p['y']) for p in highlighted_points}

    for point in frame_points:
        is_highlighted = (point['frame'], point['x'], point['y']) in highlighted_set
        center = (point['x'], point['y'])
        point_color = (0, 255, 0) if point['positive'] else (255, 0, 0)
        if is_highlighted:
            cv2.circle(image, center, 9, (0, 128, 255), 3)
        cv2.circle(image, center, 5, (255, 255, 0), 2)
        cv2.circle(image, center, 4, point_color, -1)

    return image

def apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter=None, preview_mask=None, preview_object_id=None):
    """Apply postprocessing to masks and draw them on the image for display"""
    raw_masks = core.load_masks_for_frame(
        frame_number, points, return_combined=False, object_id_filter=object_id_filter
    )

    if raw_masks:
        processed_masks = {
            object_id: core.apply_mask_postprocessing(mask) for object_id, mask in raw_masks.items()
        }
    else:
        processed_masks = {}

    # Substitute the preview mask for the selected object if provided
    if preview_mask is not None and preview_object_id is not None:
        processed_masks[preview_object_id] = core.apply_mask_postprocessing(preview_mask)

    if view_options.get("show_masks", True):
        image = draw_masks(image, processed_masks)
    if view_options.get("show_outlines", True):
        image = draw_contours(image, processed_masks)

    return image


# .........................................................................................
# Mask / session utilities
# .........................................................................................

def deduplicate_masks(parent_window):
    """Deduplicate similar masks using settings threshold"""
    settings_mgr = get_settings_manager()
    threshold = settings_mgr.app_settings.dedupe_threshold
    return replace_similar_matte_frames(parent_window, threshold)


def remove_backup_mattes():
    if os.path.exists(core.backup_dir):
        shutil.rmtree(core.backup_dir)


# .........................................................................................
# Video / image I/O
# .........................................................................................

def load_video(video_file, parent_window):
    """Load video and save frames as images using multi-threaded writers"""
    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.frames_dir)
    os.makedirs(core.mask_dir)
    os.makedirs(core.matting_dir)
    print(f"Loading video: {video_file}")

    progress_dialog = QProgressDialog("Loading video...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()

    container = av.open(video_file)
    stream = container.streams.video[0]

    # Enable threading in the decoder itself for faster demuxing
    stream.thread_type = "AUTO"

    core.VideoInfo.width = stream.width
    core.VideoInfo.height = stream.height
    core.VideoInfo.fps = float(stream.average_rate)
    # frames may be None for some containers (e.g. MKV), fall back to counting
    core.VideoInfo.total_frames = stream.frames or 0
    total_frames = core.VideoInfo.total_frames
    core.VideoInfo.color_space = src_cs = int(stream.codec_context.colorspace)  # 1=BT.709, 5=BT.601 etc.
    src_range = int(stream.codec_context.color_range)  # 1=limited, 2=full

    frame_count = 0
    settings_mgr = get_settings_manager()
    frame_format = settings_mgr.get_app_setting("frame_format", "png")

    # --- Threaded frame writing setup ---
    # A decoded 4K RGB frame is about 24 MiB.  The previous 100-frame queue could
    # therefore retain gigabytes of unified memory. PyAV decodes on its own
    # internally threaded producer while four consumers encode/write PNG frames
    # concurrently. Back-pressure is intentional: decoding pauses whenever the
    # small queue is full, keeping memory bounded on 16 GiB Macs.
    queue_size = max(1, int(settings_mgr.get_app_setting("frame_writer_queue_size", 4)))
    requested_workers = int(settings_mgr.get_app_setting("frame_writer_threads", 4))
    num_workers = max(1, min(requested_workers, os.cpu_count() or 1))
    png_compression = max(
        0, min(9, int(settings_mgr.get_app_setting("frame_png_compression", 1)))
    )
    save_q = queue.Queue(maxsize=queue_size)

    if frame_format.lower() == "png":
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    else:
        write_params = []

    def save_worker():
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done()
                break
            path, frame = item
            try:
                cv2.imwrite(path, frame, write_params)
            except Exception as e:
                print(f"Error writing {path}: {e}")
            save_q.task_done()

    writers = []
    for _ in range(num_workers):
        t = threading.Thread(target=save_worker, daemon=True)
        t.start()
        writers.append(t)

    cancelled = False

    with tqdm(total=total_frames or None) as progress:
        for frame in container.decode(stream):
            frame_rgb = frame.reformat(
                format="rgb24",
                src_colorspace=src_cs,
                dst_colorspace=1,   # always output BT.709
                src_color_range=src_range,
                dst_color_range=2,  # always output full range for PNG
            ).to_ndarray()

            # cv2.imwrite expects BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            frame_filename = os.path.join(core.frames_dir, f"{frame_count:05d}.{frame_format}")
            save_q.put((frame_filename, frame_bgr))
            frame_count += 1
            progress.update(1)

            if total_frames:
                progress_dialog.setValue(frame_count * 100 // total_frames)
            QApplication.processEvents()

            if progress_dialog.wasCanceled():
                cancelled = True
                break

    container.close()

    if cancelled:
        # Drain the queue without processing so workers can be shut down cleanly
        while not save_q.empty():
            try:
                save_q.get_nowait()
                save_q.task_done()
            except queue.Empty:
                break
        for _ in writers:
            save_q.put(None)
        for t in writers:
            t.join()
        if os.path.exists(core.temp_dir):
            shutil.rmtree(core.temp_dir)
        progress_dialog.close()
        print("Operation cancelled by user.")
        return 0

    save_q.join()
    for _ in writers:
        save_q.put(None)
    for t in writers:
        t.join()

    progress_dialog.setValue(100)
    progress_dialog.close()

    core.VideoInfo.total_frames = frame_count
    settings_mgr.set_session_setting("sequence_start_frame", 0)  # not an image sequence
    return frame_count

def detect_image_sequence(image_path):
    """
    Detect if an image is part of a sequence based on common naming patterns.
    Returns (is_sequence, sequence_files, base_name) or (False, [], None).
    base_name is the filename prefix with the trailing frame number removed
    """
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    patterns = [
        r'^(.+?)(\d{4,})$',
        r'^(.+?)(\d{3})$',
        r'^(.+?)(\d{2})$',
        r'^(.+?)_(\d+)$',
        r'^(.+?)\-(\d+)$',
        r'^(.+?)\.(\d+)$',
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            base_name = match.group(1)

            if pattern == r'^(.+?)(\d{4,})$':
                glob_pattern = f"{base_name}*{ext}"
            elif pattern == r'^(.+?)(\d{3})$':
                glob_pattern = f"{base_name}???{ext}"
            elif pattern == r'^(.+?)(\d{2})$':
                glob_pattern = f"{base_name}??{ext}"
            else:
                separator = pattern.split('(\\d+)')[0][-1]
                glob_pattern = f"{base_name}{separator}*{ext}"

            search_path = os.path.join(directory, glob_pattern)
            potential_files = glob.glob(search_path)

            sequence_files = []
            for file_path in potential_files:
                file_name = os.path.basename(file_path)
                file_base = os.path.splitext(file_name)[0]
                if re.match(pattern, file_base):
                    sequence_files.append(file_path)

            def natural_sort_key(path):
                base_name = os.path.splitext(os.path.basename(path))[0]
                match = re.match(pattern, base_name)
                if match:
                    return (match.group(1), int(match.group(2)))
                return (base_name, 0)

            sequence_files.sort(key=natural_sort_key)

            if len(sequence_files) > 1:
                return True, sequence_files, base_name

    return False, [], None


def get_sequence_start_frame(files):
    """
    Return the frame number of the first file if the files form a consecutively
    numbered sequence (each frame number is exactly one more than the previous).
    Returns 0 for single images, sequences with gaps/duplicates, or files without
    a trailing frame number.
    """
    if len(files) < 2:
        return 0

    numbers = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'(\d+)$', stem)
        if not match:
            return 0
        numbers.append(int(match.group(1)))

    first = numbers[0]
    if all(n == first + i for i, n in enumerate(numbers)):
        return first
    return 0


def load_image_sequence(image_path, parent_window):
    """
    Load an image or image sequence. Detects sequences automatically and prompts user.
    """
    is_sequence, sequence_files, sequence_base_name = detect_image_sequence(image_path)
    files_to_load = [image_path]

    if is_sequence:
        msg_box = QMessageBox(parent_window)
        msg_box.setWindowTitle("Image Sequence Detected")
        msg_box.setText(f"The selected image appears to be part of a sequence with {len(sequence_files)} images.")
        msg_box.setInformativeText("Would you like to load the entire sequence or just the single image?")

        sequence_button = msg_box.addButton("Load Sequence", QMessageBox.AcceptRole)
        single_button = msg_box.addButton("Load Single Image", QMessageBox.RejectRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.exec()

        if msg_box.clickedButton() == sequence_button:
            files_to_load = sequence_files
        elif msg_box.clickedButton() == single_button:
            files_to_load = [image_path]
        else:
            return 0

    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.frames_dir)
    os.makedirs(core.mask_dir)
    os.makedirs(core.matting_dir)

    print(f"Loading {'image sequence' if len(files_to_load) > 1 else 'image'}: {len(files_to_load)} file(s)")

    progress_dialog = QProgressDialog("Loading images...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()

    settings_mgr = get_settings_manager()
    app_frame_format = settings_mgr.get_app_setting("frame_format", "png")

    first_image = cv2.imread(files_to_load[0])
    if first_image is None:
        progress_dialog.close()
        show_message_dialog(parent_window, title="Error",
                            message=f"Could not load image: {files_to_load[0]}", type="critical")
        return 0

    core.VideoInfo.height, core.VideoInfo.width = first_image.shape[:2]
    core.VideoInfo.fps = 24.0
    core.VideoInfo.total_frames = len(files_to_load)

    for frame_count, source_path in enumerate(files_to_load):
        image = cv2.imread(source_path)
        if image is None:
            print(f"Warning: Could not load {source_path}, skipping...")
            continue

        source_ext = os.path.splitext(source_path)[1].lower()
        if source_ext in ['.png', '.jpg', '.jpeg']:
            output_ext = source_ext.lstrip('.')
            frame_filename = os.path.join(core.frames_dir, f"{frame_count:05d}.{output_ext}")
            shutil.copy2(source_path, frame_filename)
        else:
            frame_filename = os.path.join(core.frames_dir, f"{frame_count:05d}.{app_frame_format}")
            cv2.imwrite(frame_filename, image)

        progress_dialog.setValue((frame_count + 1) * 100 // len(files_to_load))
        QApplication.processEvents()

        if progress_dialog.wasCanceled():
            if os.path.exists(core.temp_dir):
                shutil.rmtree(core.temp_dir)
            progress_dialog.close()
            return 0

    progress_dialog.setValue(100)

    # Remember the source numbering so sequence exports can start from the same frame number
    settings_mgr.set_session_setting("sequence_start_frame", get_sequence_start_frame(files_to_load))

    # For an actual multi-file sequence, store a "clean" path (frame number stripped)
    # as video_file_path, so the {input_name} export filename tag doesn't end up
    # with one source frame's number baked into it.
    clean_base_name = sequence_base_name.rstrip('_-.') if sequence_base_name else sequence_base_name
    if len(files_to_load) > 1 and clean_base_name:
        clean_ext = os.path.splitext(files_to_load[0])[1]
        clean_dir = os.path.dirname(files_to_load[0])
        settings_mgr.set_session_setting("video_file_path", os.path.join(clean_dir, f"{clean_base_name}{clean_ext}"))

    return core.VideoInfo.total_frames


def resume_session():
    if os.path.exists(core.temp_dir):
        if os.path.exists(core.frames_dir) and os.listdir(core.frames_dir):
            print("Resuming previous session...")
            QApplication.processEvents()
            restore_video_info()
            return core.VideoInfo.total_frames


def restore_video_info():
    if not os.path.exists(core.frames_dir):
        return 0
    image = core.load_base_frame(0)
    if image is not None:
        height, width, channels = image.shape
        core.VideoInfo.width = width
        core.VideoInfo.height = height
        core.VideoInfo.fps = 24
        extension = core.get_frame_extension()
        core.VideoInfo.total_frames = len([f for f in os.listdir(core.frames_dir) if f.endswith(f".{extension}")])

        settings_mgr = get_settings_manager()
        if settings_mgr.session_exists():
            session_width = settings_mgr.get_session_setting("video_width", 0)
            session_height = settings_mgr.get_session_setting("video_height", 0)
            session_fps = settings_mgr.get_session_setting("video_fps", 0)
            session_frames = settings_mgr.get_session_setting("total_frames", 0)

            if session_width > 0 and session_height > 0:
                core.VideoInfo.width = session_width
                core.VideoInfo.height = session_height
            if session_fps > 0:
                core.VideoInfo.fps = session_fps
            if session_frames > 0:
                core.VideoInfo.total_frames = session_frames


def load_project(file_name, parent_window):
    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.temp_dir, exist_ok=True)
    progress = None

    try:
        with zipfile.ZipFile(file_name, 'r') as zipf:
            file_list = zipf.namelist()
            total_files = len(file_list)

        if total_files == 0:
            return True

        progress = QProgressDialog("Extracting files...", "", 0, total_files, parent_window)
        progress.setWindowTitle("Extracting Backup")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        with zipfile.ZipFile(file_name, 'r') as zipf:
            for i, file_name in enumerate(file_list):
                progress.setValue(i)
                progress.setLabelText(f"Extracting: {os.path.basename(file_name)}")
                QApplication.processEvents()
                zipf.extract(file_name, core.temp_dir)

        progress.setValue(total_files)
        progress.setLabelText("Extraction completed!")
        QApplication.processEvents()
        return True

    except Exception as e:
        if progress:
            progress.close()
        raise e

    finally:
        if progress:
            progress.close()


def save_project(file_name, parent_window):
    total_files = 0
    all_files = []

    for root, dirs, files in os.walk(core.temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, core.temp_dir)
            all_files.append((file_path, arcname))
            total_files += 1

    progress = QProgressDialog("Backing up files...", "Cancel", 0, total_files, parent_window)
    progress.setWindowTitle("Creating Backup")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()
    QApplication.processEvents()

    try:
        with zipfile.ZipFile(file_name, 'w', zipfile.ZIP_STORED) as zipf:
            for i, (file_path, arcname) in enumerate(all_files):
                if progress.wasCanceled():
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                    return 0

                progress.setValue(i)
                progress.setLabelText(f"Adding: {os.path.basename(file_path)}")
                QApplication.processEvents()
                zipf.write(file_path, arcname)

        progress.setValue(total_files)
        QApplication.processEvents()

    except Exception as e:
        progress.close()
        try:
            os.remove(file_name)
        except Exception:
            pass
        raise e

    finally:
        if not progress.wasCanceled():
            progress.close()

    return 1
