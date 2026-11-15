from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import platform
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from sammie.segmentation.mlx_sam31 import (
    SAM31_MODEL_NAME,
    Sam31MLXPredictor,
    _clone_containers,
    clear_mlx_sam31_cache,
    is_mlx_sam31_model,
    sam31_checkpoint_ready,
)


class FakeMX:
    float32 = np.float32
    int32 = np.int32

    @staticmethod
    def array(value, dtype=None):
        return np.asarray(value, dtype=dtype)


class Sam31AdapterTests(unittest.TestCase):
    def test_model_name_is_explicit(self):
        self.assertTrue(is_mlx_sam31_model(SAM31_MODEL_NAME))
        self.assertFalse(is_mlx_sam31_model("Large"))

    def test_points_are_scaled_to_native_1008_coordinates(self):
        predictor = object.__new__(Sam31MLXPredictor)
        predictor.mx = FakeMX()
        predictor.image_size = 1008
        result = predictor._scaled_points(
            {"video_width": 2016, "video_height": 504},
            np.array([[1008, 252]], dtype=np.float32),
            np.array([1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            result["point_coords"], np.array([[[504, 504]]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result["point_labels"], np.array([[1]], dtype=np.int32)
        )

    def test_preview_container_clone_does_not_leak_mutation(self):
        original = {
            "output": {0: {"conditioning_objects": {0}, "items": [1, 2]}},
            "tensor": np.array([1, 2, 3]),
        }
        snapshot = _clone_containers(original)
        original["output"][0]["conditioning_objects"].add(1)
        original["output"][0]["items"].append(3)

        self.assertEqual(snapshot["output"][0]["conditioning_objects"], {0})
        self.assertEqual(snapshot["output"][0]["items"], [1, 2])
        self.assertIs(snapshot["tensor"], original["tensor"])

    def test_reverse_memory_pruning_keeps_only_native_horizon(self):
        predictor = object.__new__(Sam31MLXPredictor)
        tracker_model = type(
            "TrackerModel",
            (),
            {
                "max_obj_ptrs_in_encoder": 16,
                "num_maskmem": 7,
                "memory_temporal_stride_for_eval": 1,
            },
        )()
        predictor.model = type("Model", (), {"tracker_model": tracker_model})()
        tracker_state = type(
            "TrackerState",
            (),
            {"non_cond_frame_outputs": {i: object() for i in range(50)}},
        )()
        state = {
            "_tracker_state": tracker_state,
            "output_dict_per_obj": {
                0: {
                    "non_cond_frame_outputs": {i: object() for i in range(50)}
                }
            },
        }

        predictor._prune_non_conditioning_memory(
            state, frame_idx=10, reverse=True
        )
        expected = list(range(10, 27))
        self.assertEqual(sorted(tracker_state.non_cond_frame_outputs), expected)
        self.assertEqual(
            sorted(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]),
            expected,
        )

    def test_prompt_correction_discards_stale_propagated_timeline(self):
        predictor = object.__new__(Sam31MLXPredictor)
        tracker_state = type(
            "TrackerState",
            (),
            {"non_cond_frame_outputs": {242: object(), 313: object()}},
        )()
        state = {
            "_tracker_state": tracker_state,
            "output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {241: object()},
                    "non_cond_frame_outputs": {242: object(), 313: object()},
                }
            },
            "temp_output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {250: object()},
                }
            },
            "frames_tracked_per_obj": {
                0: {241: {"reverse": False}, 313: {"reverse": False}}
            },
        }

        predictor._discard_non_conditioning_memory(state)

        self.assertEqual(tracker_state.non_cond_frame_outputs, {})
        self.assertEqual(
            state["output_dict_per_obj"][0]["non_cond_frame_outputs"], {}
        )
        self.assertEqual(
            state["temp_output_dict_per_obj"][0]["non_cond_frame_outputs"], {}
        )
        self.assertEqual(state["frames_tracked_per_obj"][0], {})
        self.assertIn(241, state["output_dict_per_obj"][0]["cond_frame_outputs"])

    @unittest.skipUnless(
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and sam31_checkpoint_ready(),
        "requires the local SAM 3.1 MLX checkpoint on Apple Silicon",
    )
    def test_worker_loaded_predictor_can_prompt_and_track_on_caller_thread(self):
        with TemporaryDirectory() as directory:
            for frame in range(2):
                Image.new(
                    "RGB", (32, 32), color=(20 + frame, 40, 60)
                ).save(f"{directory}/{frame:05d}.png")
            with ThreadPoolExecutor(max_workers=1) as pool:
                predictor = pool.submit(Sam31MLXPredictor).result()

            state = predictor.init_state(directory)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=0,
                points=np.array([[16, 16]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
            tracked = list(
                predictor.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=1
                )
            )

            self.assertEqual([frame for frame, _, _ in tracked], [0, 1])
            del state, predictor
            clear_mlx_sam31_cache()


if __name__ == "__main__":
    unittest.main()
