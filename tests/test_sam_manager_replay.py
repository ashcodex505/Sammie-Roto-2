import unittest
from unittest.mock import patch

import numpy as np

from sammie import core
from sammie.sammie import SamManager


class _FakePredictor:
    def __init__(self):
        self.calls = []

    def reset_state(self, inference_state):
        self.calls.append(("reset", inference_state))

    def add_new_points_or_box(
        self, *, inference_state, frame_idx, obj_id, points, labels,
        clear_old_points,
    ):
        self.calls.append(
            ("prompt", frame_idx, obj_id, points.copy(), labels.copy())
        )
        return frame_idx, [obj_id], [np.ones((1, 2, 2), dtype=np.float32)]

    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=False
    ):
        self.calls.append(("clear", frame_idx, obj_id, need_output))
        obj_idx = inference_state["obj_id_to_idx"][obj_id]
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)


class SamManagerReplayTests(unittest.TestCase):
    def _manager(self):
        manager = SamManager()
        manager.predictor = _FakePredictor()
        manager.inference_state = {"state": True}
        manager.loaded_model_name = "Base+ MLX FP16"
        return manager

    @patch("sammie.sammie.cv2.imwrite", return_value=True)
    @patch("sammie.sammie.os.makedirs")
    def test_replay_batches_all_points_for_each_object_and_frame(
        self, _makedirs, _imwrite
    ):
        manager = self._manager()
        points = [
            {"frame": 2, "object_id": 0, "x": 10, "y": 20, "positive": True},
            {"frame": 2, "object_id": 0, "x": 30, "y": 40, "positive": False},
            {"frame": 2, "object_id": 0, "x": 50, "y": 60, "positive": True},
            {"frame": 2, "object_id": 1, "x": 70, "y": 80, "positive": True},
            {"frame": 4, "object_id": 1, "x": 90, "y": 100, "positive": True},
        ]

        with patch.object(core.VideoInfo, "total_frames", 6):
            manager.replay_points(points)

        prompt_calls = [call for call in manager.predictor.calls if call[0] == "prompt"]
        self.assertEqual(len(prompt_calls), 3)
        self.assertEqual(
            [(call[1], call[2], len(call[3])) for call in prompt_calls],
            [(2, 0, 3), (2, 1, 1), (4, 1, 1)],
        )
        np.testing.assert_array_equal(prompt_calls[0][4], [1, 0, 1])

    @patch("sammie.sammie.cv2.imwrite", return_value=True)
    @patch("sammie.sammie.os.makedirs")
    def test_replay_order_is_stable_by_frame_then_object(self, _makedirs, _imwrite):
        manager = self._manager()
        points = [
            {"frame": 3, "object_id": 8, "x": 1, "y": 1, "positive": True},
            {"frame": 1, "object_id": 2, "x": 2, "y": 2, "positive": True},
            {"frame": 1, "object_id": 0, "x": 3, "y": 3, "positive": True},
        ]

        with patch.object(core.VideoInfo, "total_frames", 5):
            manager.replay_points(points)

        prompts = [call for call in manager.predictor.calls if call[0] == "prompt"]
        self.assertEqual([(call[1], call[2]) for call in prompts], [(1, 0), (1, 2), (3, 8)])

    @patch("sammie.sammie.cv2.imwrite", return_value=True)
    @patch("sammie.sammie.os.makedirs")
    def test_replace_points_updates_only_one_frame_and_object(
        self, _makedirs, _imwrite
    ):
        manager = self._manager()
        other_output = object()
        manager.inference_state = {
            "obj_id_to_idx": {3: 0, 4: 1},
            "point_inputs_per_obj": {0: {8: object()}, 1: {9: object()}},
            "output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {8: object()},
                    "non_cond_frame_outputs": {},
                },
                1: {
                    "cond_frame_outputs": {9: other_output},
                    "non_cond_frame_outputs": {},
                },
            },
            "temp_output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {8: object()},
                    "non_cond_frame_outputs": {},
                },
                1: {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {},
                },
            },
            "frames_tracked_per_obj": {
                0: {8: {"reverse": False}},
                1: {9: {"reverse": False}},
            },
        }
        remaining = [
            {"frame": 8, "object_id": 3, "x": 20, "y": 30, "positive": False}
        ]

        manager.replace_frame_object_points(8, 3, remaining)

        self.assertNotIn(
            "reset", [call[0] for call in manager.predictor.calls]
        )
        self.assertEqual(
            [(call[0], call[1], call[2]) for call in manager.predictor.calls],
            [("clear", 8, 3), ("prompt", 8, 3)],
        )
        self.assertIs(
            manager.inference_state["output_dict_per_obj"][1]
            ["cond_frame_outputs"][9],
            other_output,
        )

    def test_replace_with_no_points_clears_without_model_inference(self):
        manager = self._manager()
        manager.inference_state = {
            "obj_id_to_idx": {3: 0},
            "point_inputs_per_obj": {0: {8: object()}},
            "output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {8: object()},
                    "non_cond_frame_outputs": {},
                }
            },
            "temp_output_dict_per_obj": {
                0: {
                    "cond_frame_outputs": {8: object()},
                    "non_cond_frame_outputs": {},
                }
            },
            "frames_tracked_per_obj": {0: {8: {"reverse": False}}},
        }

        manager.replace_frame_object_points(8, 3, [])

        self.assertEqual(
            manager.predictor.calls,
            [("clear", 8, 3, False)],
        )
        self.assertEqual(
            manager.inference_state["output_dict_per_obj"][0]
            ["cond_frame_outputs"],
            {},
        )


if __name__ == "__main__":
    unittest.main()
