from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sammie.segmentation.mlx_sam21 import (
    LazyFrameSequence,
    MLX_MODEL_NAME,
    _predict_initial_frame,
    _prune_non_conditioning_state,
    is_mlx_sam21_model,
)


class LazyFrameSequenceTests(unittest.TestCase):
    def test_cache_is_bounded_and_slices_match_predictor_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                pixels = np.full((10, 20, 3), index * 10, dtype=np.uint8)
                Image.fromarray(pixels).save(root / f"{index:05d}.png")

            frames = LazyFrameSequence(root, image_size=16, cache_frames=2)
            self.assertEqual(frames.shape, (6, 3, 16, 16))
            self.assertEqual(frames.video_width, 20)
            self.assertEqual(frames.video_height, 10)
            self.assertEqual(frames[0:1].shape, (1, 3, 16, 16))
            self.assertEqual(frames[1:4].shape, (3, 3, 16, 16))
            self.assertLessEqual(len(frames._cache), 2)
            self.assertLessEqual(frames.nbytes, 2 * 3 * 16 * 16 * 2)


class StatePruningTests(unittest.TestCase):
    def test_keeps_conditioning_frames_and_exact_forward_horizon(self):
        state = {
            "output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {0: object(), 5: object()},
                    "non_cond_frame_outputs": {i: object() for i in range(40)},
                }
            }
        }
        _prune_non_conditioning_state(state, current_frame=39, reverse=False, window=16)
        output = state["output_dict_per_obj"][0]
        self.assertEqual(sorted(output["cond_frame_outputs"]), [0, 5])
        self.assertEqual(sorted(output["non_cond_frame_outputs"]), list(range(24, 40)))

    def test_keeps_reverse_horizon(self):
        state = {
            "output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {39: object()},
                    "non_cond_frame_outputs": {i: object() for i in range(40)},
                }
            }
        }
        _prune_non_conditioning_state(state, current_frame=0, reverse=True, window=16)
        self.assertEqual(
            sorted(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]),
            list(range(16)),
        )

    def test_model_name_is_explicit(self):
        self.assertTrue(is_mlx_sam21_model(MLX_MODEL_NAME))
        self.assertFalse(is_mlx_sam21_model("Base"))


class InitialFrameRefinementTests(unittest.TestCase):
    def test_second_click_uses_previous_mask_without_temporal_memory(self):
        class FakeMX:
            @staticmethod
            def array(value):
                return np.asarray(value)

        class FakeModel:
            def __init__(self):
                self.call = None

            def predict_from_encoded(self, encoded, coords, labels, **kwargs):
                self.call = (encoded, coords, labels, kwargs)
                return {"refined": True}

        model = FakeModel()
        prior_mask = np.ones((1, 1, 256, 256), dtype=np.float32)
        prompts = {
            "point_coords": np.array([[10, 20], [30, 40]], dtype=np.float32),
            "point_labels": np.array([1, 0], dtype=np.int32),
        }

        out = _predict_initial_frame(
            model,
            {"vision_features": "encoded"},
            prompts,
            prior_mask,
            lambda *_: self.fail("refinement must not restart initial prediction"),
            FakeMX,
        )

        self.assertEqual(out, {"refined": True})
        self.assertIs(model.call[3]["mask_input"], prior_mask)
        self.assertFalse(model.call[3]["multimask_output"])
        self.assertTrue(model.call[3]["add_no_mem_embed"])
        self.assertEqual(model.call[1].shape, (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
