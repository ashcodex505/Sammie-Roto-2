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


class SamManagerReplayTests(unittest.TestCase):
    def _manager(self):
        manager = SamManager()
        manager.predictor = _FakePredictor()
        manager.inference_state = {"state": True}
        manager.loaded_model_name = "SAM 3.1 MLX"
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


if __name__ == "__main__":
    unittest.main()
