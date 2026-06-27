import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

from app.main import _load_damage_segformer_model, _load_parts_segmentation_model


class DamageModelLoadingTests(unittest.TestCase):
    @patch("app.main.YOLO")
    def test_parts_segmentation_checkpoint_is_accepted(self, yolo):
        yolo.return_value.task = "segment"

        self.assertIs(_load_parts_segmentation_model("parts.pt"), yolo.return_value)

    @patch("app.main.YOLO")
    def test_parts_detection_checkpoint_is_rejected(self, yolo):
        yolo.return_value.task = "detect"

        with self.assertRaisesRegex(ValueError, "car-parts segmentation checkpoint"):
            _load_parts_segmentation_model("parts.pt")

    @patch("torch.cuda.is_available", return_value=False)
    def test_damage_segformer_model_is_loaded(self, _cuda):
        model = MagicMock()
        model.config.id2label = {"0": "background", "1": "scratch"}
        processor = object()
        fake_transformers = SimpleNamespace(
            SegformerForSemanticSegmentation=SimpleNamespace(from_pretrained=MagicMock(return_value=model)),
            SegformerImageProcessor=SimpleNamespace(from_pretrained=MagicMock(return_value=processor)),
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            bundle = _load_damage_segformer_model("damage-segformer")

        self.assertIs(bundle.model, model)
        self.assertIs(bundle.processor, processor)
        self.assertEqual(bundle.device, "cpu")
        self.assertEqual(bundle.id2label, {0: "background", 1: "scratch"})
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
