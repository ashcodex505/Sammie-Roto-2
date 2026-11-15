from __future__ import annotations

import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QPushButton,
    QSplitter,
    QWidget,
)

from sammie import core
from sammie.gui_widgets import PointTable
from sammie_main import MainWindow


class _PointManager:
    def __init__(self, points):
        self.points = list(points)

    def remove_point(self, frame, object_id, x, y):
        for index, point in enumerate(self.points):
            if (
                point["frame"] == frame
                and point["object_id"] == object_id
                and point["x"] == x
                and point["y"] == y
            ):
                return self.points.pop(index)
        return None

    def get_all_points(self):
        return list(self.points)


class _ReplayManager:
    def __init__(self):
        self.propagated = True
        self.replayed_points = None

    def replay_points(self, points):
        self.replayed_points = points


class _ParentWindow:
    def __init__(self, points):
        self.point_manager = _PointManager(points)
        self.sam_manager = _ReplayManager()
        self.matany_manager = _ReplayManager()
        self.tracking_status_updates = 0

    def update_tracking_status(self):
        self.tracking_status_updates += 1


class PointTableSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_uses_extended_row_selection(self):
        table = PointTable()
        self.assertEqual(
            table.selectionMode(), QAbstractItemView.ExtendedSelection
        )
        self.assertEqual(table.selectionBehavior(), QAbstractItemView.SelectRows)

    def test_bulk_delete_removes_every_selected_row_and_replays_once(self):
        points = [
            {"frame": 10, "object_id": 1, "positive": True, "x": 100, "y": 200},
            {"frame": 11, "object_id": 1, "positive": True, "x": 110, "y": 210},
            {"frame": 12, "object_id": 2, "positive": False, "x": 120, "y": 220},
        ]
        table = PointTable()
        parent = _ParentWindow(points)
        table.parent_window = parent
        for point in points:
            table.add_point(**point)

        selection_counts = []
        table.selection_count_changed.connect(selection_counts.append)
        flags = QItemSelectionModel.Select | QItemSelectionModel.Rows
        table.selectionModel().select(table.model().index(0, 0), flags)
        table.selectionModel().select(table.model().index(2, 0), flags)

        with TemporaryDirectory() as mask_directory, patch.object(
            core, "mask_dir", mask_directory
        ):
            table.delete_selected_row()

        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(parent.point_manager.get_all_points(), [points[1]])
        self.assertEqual(parent.sam_manager.replayed_points, [points[1]])
        self.assertFalse(parent.sam_manager.propagated)
        self.assertFalse(parent.matany_manager.propagated)
        self.assertEqual(parent.tracking_status_updates, 1)
        self.assertEqual(selection_counts[-1], 0)

    def test_expand_and_restore_point_list_panels(self):
        window = type("WindowParts", (), {})()
        window.vertical_splitter = QSplitter(Qt.Vertical)
        window.center_panel = QWidget()
        bottom_panel = QWidget()
        window.vertical_splitter.addWidget(window.center_panel)
        window.vertical_splitter.addWidget(bottom_panel)
        window.vertical_splitter.resize(800, 600)
        window.vertical_splitter.setSizes([400, 200])

        window.bottom_splitter = QSplitter(Qt.Horizontal)
        point_panel = QWidget()
        window.console_container = QWidget()
        window.bottom_splitter.addWidget(point_panel)
        window.bottom_splitter.addWidget(window.console_container)
        window.bottom_splitter.resize(800, 200)
        window.bottom_splitter.setSizes([350, 450])

        window.expand_point_list_btn = QPushButton("Expand")
        original_vertical_sizes = window.vertical_splitter.sizes()
        original_bottom_sizes = window.bottom_splitter.sizes()

        MainWindow._toggle_point_list_expanded(window, True)
        self.assertTrue(window.center_panel.isHidden())
        self.assertTrue(window.console_container.isHidden())
        self.assertEqual(window.expand_point_list_btn.text(), "Restore")

        MainWindow._toggle_point_list_expanded(window, False)
        self.assertFalse(window.center_panel.isHidden())
        self.assertFalse(window.console_container.isHidden())
        self.assertEqual(window.expand_point_list_btn.text(), "Expand")
        self.assertEqual(window.vertical_splitter.sizes(), original_vertical_sizes)
        self.assertEqual(window.bottom_splitter.sizes(), original_bottom_sizes)


if __name__ == "__main__":
    unittest.main()
