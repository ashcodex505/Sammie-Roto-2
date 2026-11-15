import unittest

from sammie.core import next_available_object_id


class NextAvailableObjectIdTests(unittest.TestCase):
    def test_returns_lowest_unused_id(self):
        points = [
            {"object_id": 0},
            {"object_id": 0},
            {"object_id": 2},
            {"object_id": 8},
        ]
        self.assertEqual(next_available_object_id(points, maximum=20), 1)

    def test_returns_none_when_all_ids_are_used(self):
        points = [{"object_id": object_id} for object_id in range(3)]
        self.assertIsNone(next_available_object_id(points, maximum=2))


if __name__ == "__main__":
    unittest.main()
