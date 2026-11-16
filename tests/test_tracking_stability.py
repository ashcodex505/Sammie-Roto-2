import unittest

import numpy as np

from sammie.sammie import (
    _released_foreground_region,
    _stabilize_propagated_mask,
)


class TrackingStabilityTests(unittest.TestCase):
    def test_trims_a_connected_leak_beyond_the_previous_mask_envelope(self):
        previous = np.zeros((32, 32), dtype=np.uint8)
        previous[10:20, 8:18] = 255

        predicted = previous.copy()
        predicted[14:17, 18:30] = 255

        stabilized = _stabilize_propagated_mask(
            predicted, previous, motion_radius=2
        )

        self.assertTrue(np.all(stabilized[10:20, 8:18] == 255))
        self.assertEqual(np.count_nonzero(stabilized[:, 21:]), 0)

    def test_preserves_normal_frame_to_frame_motion(self):
        previous = np.zeros((32, 32), dtype=np.uint8)
        previous[10:20, 8:18] = 255
        shifted = np.zeros_like(previous)
        shifted[10:20, 10:20] = 255

        stabilized = _stabilize_propagated_mask(
            shifted, previous, motion_radius=2
        )

        np.testing.assert_array_equal(stabilized, shifted)

    def test_does_not_lock_a_disappeared_region_in_place(self):
        previous = np.zeros((32, 32), dtype=np.uint8)
        previous[8:24, 6:26] = 255
        collapsed = np.zeros_like(previous)
        collapsed[8:13, 6:26] = 255

        stabilized = _stabilize_propagated_mask(
            collapsed, previous, motion_radius=2
        )

        np.testing.assert_array_equal(stabilized, collapsed)

    def test_returns_prediction_when_guard_would_remove_most_of_it(self):
        previous = np.zeros((32, 32), dtype=np.uint8)
        previous[2:6, 2:6] = 255
        new_location = np.zeros_like(previous)
        new_location[20:28, 20:28] = 255

        stabilized = _stabilize_propagated_mask(
            new_location, previous, motion_radius=2
        )

        np.testing.assert_array_equal(stabilized, new_location)

    def test_reclaims_window_pixels_revealed_by_a_moving_foreground(self):
        window = np.zeros((32, 32), dtype=np.uint8)
        window[5:27, 5:14] = 255
        old_foreground = np.zeros_like(window)
        old_foreground[5:27, 14:20] = 255
        new_foreground = np.zeros_like(window)
        new_foreground[5:27, 18:24] = 255

        released = _released_foreground_region(
            window,
            old_foreground,
            new_foreground,
            motion_radius=6,
        )

        self.assertTrue(np.all(released[5:27, 14:18] == 255))
        self.assertEqual(np.count_nonzero(released[:, 18:]), 0)


if __name__ == "__main__":
    unittest.main()
