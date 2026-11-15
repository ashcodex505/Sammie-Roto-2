from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from sammie.gui_widgets import ConsoleRedirect
from sammie_main import MainWindow


class ConsoleRedirectTests(unittest.TestCase):
    def test_full_log_is_preserved_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                emitted = []
                redirect = ConsoleRedirect()
                redirect.text_written.connect(
                    lambda text, carriage_return: emitted.append(
                        (text, carriage_return)
                    )
                )

                redirect.write("\rtracking 50%")
                redirect.write("tracking complete\n")
                redirect.flush()
                redirect.close()
                redirect.close()

                log_bytes = Path("sammie_debug.log").read_bytes()
                log = log_bytes.decode("utf-8")
                self.assertIn(b"\rtracking 50%tracking complete\n", log_bytes)
                self.assertEqual(emitted[0], ("tracking 50%", True))
                self.assertEqual(emitted[1], ("tracking complete\n", False))
                self.assertEqual(log.count("=== Ended:"), 1)
            finally:
                os.chdir(previous_directory)


class UnloadSegmentationControlTests(unittest.TestCase):
    def test_unload_keeps_ui_reload_path_available(self):
        class FakeButton:
            def __init__(self):
                self.enabled = None
                self.text = None

            def setEnabled(self, enabled):
                self.enabled = enabled

            def setText(self, text):
                self.text = text

        class FakeManager:
            predictor = object()
            loaded_model_name = "SAM 3.1 MLX"

            def __init__(self):
                self.unloaded = False

            def unload_segmentation_model(self):
                self.unloaded = True
                self.predictor = None
                self.loaded_model_name = None

        fake = type("FakeWindow", (), {})()
        fake.sam_manager = FakeManager()
        fake.segmentation_tab = type(
            "SegmentationTab", (), {"sam_model_btn": FakeButton()}
        )()
        fake.matting_tab = type(
            "MattingTab", (), {"unload_segmentation_btn": FakeButton()}
        )()

        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.Yes
        ):
            MainWindow.unload_segmentation_for_matting(fake)

        self.assertTrue(fake.sam_manager.unloaded)
        self.assertTrue(fake.segmentation_tab.sam_model_btn.enabled)
        self.assertFalse(fake.matting_tab.unload_segmentation_btn.enabled)
        self.assertEqual(
            fake.matting_tab.unload_segmentation_btn.text,
            "SAM Unloaded",
        )


if __name__ == "__main__":
    unittest.main()
