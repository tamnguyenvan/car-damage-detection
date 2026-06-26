import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

from app.main import (
    SegmentationPrediction,
    _assessment_detections,
    _segmentation_geometry_from_result,
    match_damage_to_part,
)


def prediction(name: str, mask: np.ndarray | None, polygon=None) -> SegmentationPrediction:
    return SegmentationPrediction(
        box=[10, 10, 60, 60],
        confidence=0.92,
        class_id=8,
        class_name=name,
        mask=mask,
        polygon=polygon,
    )


class PartMatchingTests(unittest.TestCase):
    def test_damage_mask_coverage_assigns_expected_part_and_returns_iou(self):
        damage_mask = np.zeros((100, 100), dtype=np.uint8)
        damage_mask[20:40, 20:40] = 1
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[10:60, 10:60] = 1

        matched_part, coverage, iou = match_damage_to_part(
            damage_mask,
            [prediction("front_bumper", part_mask)],
            (100, 100, 3),
            threshold=0.5,
        )

        self.assertEqual(matched_part.class_name, "front_bumper")
        self.assertEqual(coverage, 1.0)
        self.assertAlmostEqual(iou, 400 / 2500)

    def test_coverage_below_threshold_returns_no_match(self):
        damage_mask = np.zeros((100, 100), dtype=np.uint8)
        damage_mask[20:40, 20:40] = 1
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[20:25, 20:25] = 1

        matched_part, coverage, iou = match_damage_to_part(
            damage_mask,
            [prediction("hood", part_mask)],
            (100, 100, 3),
            threshold=0.5,
        )

        self.assertIsNone(matched_part)
        self.assertIsNone(coverage)
        self.assertIsNone(iou)

    def test_missing_damage_mask_never_falls_back_to_box_matching(self):
        part_mask = np.ones((100, 100), dtype=np.uint8)

        matched_part, coverage, iou = match_damage_to_part(
            None,
            [prediction("wheel", part_mask)],
            (100, 100, 3),
        )

        self.assertIsNone(matched_part)
        self.assertIsNone(coverage)
        self.assertIsNone(iou)

    def test_no_parts_returns_no_match(self):
        damage_mask = np.ones((100, 100), dtype=np.uint8)
        matched_part, coverage, iou = match_damage_to_part(damage_mask, [], (100, 100, 3))

        self.assertIsNone(matched_part)
        self.assertIsNone(coverage)
        self.assertIsNone(iou)

    def test_segmentation_geometry_preserves_polygon(self):
        result = SimpleNamespace(
            masks=SimpleNamespace(
                xy=[np.array([[10.5, 10.5], [40.5, 10.5], [40.5, 40.5]], dtype=np.float32)]
            )
        )

        masks, polygons = _segmentation_geometry_from_result(result, 1, (100, 100, 3))

        self.assertEqual(masks[0][20, 20], 1)
        self.assertEqual(polygons[0], [[10.5, 10.5], [40.5, 10.5], [40.5, 40.5]])

    def test_assessment_returns_damage_and_part_polygons(self):
        damage_mask = np.zeros((100, 100), dtype=np.uint8)
        damage_mask[20:40, 20:40] = 1
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[10:60, 10:60] = 1
        damage = prediction("dent", damage_mask, [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0]])
        part = prediction("front_bumper", part_mask, [[10.0, 10.0], [60.0, 10.0], [60.0, 60.0]])

        detections = _assessment_detections([damage], [part], (100, 100, 3))

        self.assertEqual(detections[0].damage_polygon, damage.polygon)
        self.assertEqual(detections[0].car_part, "front_bumper")
        self.assertEqual(detections[0].car_part_polygon, part.polygon)
        self.assertEqual(detections[0].part_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
