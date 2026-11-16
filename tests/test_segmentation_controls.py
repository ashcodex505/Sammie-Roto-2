import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSpinBox

from sammie_main import SegmentationTab


class _Settings:
    def __init__(self):
        self.values = {"holes": 50, "dots": 50}

    def get_session_setting(self, key, default=None):
        return self.values.get(key, default)

    def set_session_setting(self, key, value):
        self.values[key] = value


class SegmentationCleanupControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_cleanup_strength_supports_4k_hole_values(self):
        settings = _Settings()
        with patch("sammie_main.get_settings_manager", return_value=settings):
            tab = SegmentationTab()
            self.assertEqual(tab.holes_slider.maximum(), 500)
            self.assertEqual(tab.dots_slider.maximum(), 200)
            self.assertIsInstance(tab.holes_value, QSpinBox)
            self.assertIsInstance(tab.dots_value, QSpinBox)

            tab.holes_value.setValue(400)
            tab.dots_value.setValue(137)

            self.assertEqual(tab.holes_slider.value(), 400)
            self.assertEqual(tab.dots_slider.value(), 137)
            self.assertEqual(settings.values["holes"], 400)
            self.assertEqual(settings.values["dots"], 137)

    def test_precision_tracking_controls_default_on_and_persist(self):
        settings = _Settings()
        with patch("sammie_main.get_settings_manager", return_value=settings):
            tab = SegmentationTab()

            self.assertTrue(tab.selected_object_only_checkbox.isChecked())
            self.assertTrue(tab.prevent_mask_drift_checkbox.isChecked())

            tab.selected_object_only_checkbox.setChecked(False)
            tab.prevent_mask_drift_checkbox.setChecked(False)

            self.assertFalse(settings.values["track_selected_object_only"])
            self.assertFalse(settings.values["prevent_mask_drift"])


if __name__ == "__main__":
    unittest.main()
