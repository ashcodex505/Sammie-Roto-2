import cv2
import numpy as np
import unittest

from sammie.core import fill_small_holes, remove_small_dots


class MaskPostprocessingTests(unittest.TestCase):
    def test_fill_small_holes_uses_exact_pixel_area_and_extended_strength(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[20:280, 20:280] = 255
        mask[80:180, 90:220] = 0  # 13,000-pixel enclosed hole

        below_threshold = fill_small_holes(mask, 110)
        filled = fill_small_holes(mask, 120)

        self.assertEqual(below_threshold[100, 100], 0)
        self.assertEqual(filled[100, 100], 255)

    def test_fill_small_holes_does_not_fill_background_connected_to_frame_edge(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:180, 20:180] = 255
        mask[80:120, 0:100] = 0

        filled = fill_small_holes(mask, 200)

        self.assertEqual(filled[100, 50], 0)

    def test_remove_small_dots_preserves_retained_mask_opacity(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[20:220, 20:220] = 255
        mask[20:220, 20] = 96
        mask[240:290, 240:290] = 180  # 2,500-pixel isolated component

        cleaned = remove_small_dots(mask, 60)

        self.assertEqual(cleaned[100, 20], 96)
        self.assertEqual(cleaned[250, 250], 0)

    def test_remove_small_dots_keeps_components_above_threshold(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[20:220, 20:220] = 255
        cv2.circle(mask, (260, 260), 35, 200, thickness=cv2.FILLED)

        cleaned = remove_small_dots(mask, 50)

        self.assertEqual(cleaned[260, 260], 200)


if __name__ == "__main__":
    unittest.main()
