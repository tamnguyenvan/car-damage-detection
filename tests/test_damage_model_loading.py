import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

from app.main import _load_segmentation_model


class DamageModelLoadingTests(unittest.TestCase):
    @patch("app.main.YOLO")
    def test_segmentation_checkpoint_is_accepted(self, yolo):
        yolo.return_value.task = "segment"

        self.assertIs(_load_segmentation_model("damage.pt", "damage"), yolo.return_value)

    @patch("app.main.YOLO")
    def test_detection_checkpoint_is_rejected(self, yolo):
        yolo.return_value.task = "detect"

        with self.assertRaisesRegex(ValueError, "damage segmentation checkpoint"):
            _load_segmentation_model("damage.pt", "damage")


if __name__ == "__main__":
    unittest.main()
