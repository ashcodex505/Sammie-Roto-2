import errno
import unittest
from unittest.mock import patch

from sammie.export_dialog import ExportPathManager


class ExportPathPermissionTests(unittest.TestCase):
    def test_reports_existing_output_locked_by_another_application(self):
        output_path = "/tmp/already-open.mov"
        permission_error = PermissionError(
            errno.EPERM, "Operation not permitted", output_path
        )

        with patch("sammie.export_dialog.os.path.exists", return_value=True), patch(
            "sammie.export_dialog.os.open", side_effect=permission_error
        ):
            message = ExportPathManager.get_write_error(output_path)

        self.assertIn("another application", message)
        self.assertIn("already-open.mov", message)


if __name__ == "__main__":
    unittest.main()
