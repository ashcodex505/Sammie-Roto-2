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

    def test_cleanup_strength_supports_exact_values_through_200(self):
        settings = _Settings()
        with patch("sammie_main.get_settings_manager", return_value=settings):
            tab = SegmentationTab()
            self.assertEqual(tab.holes_slider.maximum(), 200)
            self.assertEqual(tab.dots_slider.maximum(), 200)
            self.assertIsInstance(tab.holes_value, QSpinBox)
            self.assertIsInstance(tab.dots_value, QSpinBox)

            tab.holes_value.setValue(121)
            tab.dots_value.setValue(137)

            self.assertEqual(tab.holes_slider.value(), 121)
            self.assertEqual(tab.dots_slider.value(), 137)
            self.assertEqual(settings.values["holes"], 121)
            self.assertEqual(settings.values["dots"], 137)


if __name__ == "__main__":
    unittest.main()
