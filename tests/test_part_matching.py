import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

from app.main import (
    SegmentationPrediction,
    _assessment_detections,
    _damage_roi_from_parts,
    _extract_damage_predictions_from_semantic_map,
    _project_damage_predictions_to_image,
    _segmentation_geometry_from_result,
    match_damage_to_part,
)


def prediction(name: str, mask: np.ndarray | None, polygon=None, class_id: int = 8) -> SegmentationPrediction:
    return SegmentationPrediction(
        box=[10, 10, 60, 60],
        confidence=0.92,
        class_id=class_id,
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

    def test_assessment_merges_same_damage_on_same_part_and_pluralizes_label(self):
        scratch_a = np.zeros((100, 100), dtype=np.uint8)
        scratch_a[20:30, 20:30] = 1
        scratch_b = np.zeros((100, 100), dtype=np.uint8)
        scratch_b[35:45, 40:50] = 1
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[10:60, 10:60] = 1

        detections = _assessment_detections(
            [
                prediction("scratch", scratch_a, class_id=5),
                prediction("scratch", scratch_b, class_id=5),
            ],
            [prediction("left-fender", part_mask)],
            (100, 100, 3),
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "scratch")
        self.assertEqual(detections[0].damage_label, "scratches")
        self.assertEqual(detections[0].damage_count, 2)
        self.assertEqual(detections[0].display_label, "left-fender: scratches")
        self.assertEqual(detections[0].car_part, "left-fender")
        self.assertEqual(detections[0].box, [20.0, 20.0, 50.0, 45.0])
        self.assertEqual(detections[0].part_coverage, 1.0)

    def test_assessment_suppresses_scratch_when_dent_exists_on_same_part(self):
        scratch_mask = np.zeros((100, 100), dtype=np.uint8)
        scratch_mask[20:30, 20:30] = 1
        dent_mask = np.zeros((100, 100), dtype=np.uint8)
        dent_mask[35:50, 35:50] = 1
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[10:60, 10:60] = 1

        detections = _assessment_detections(
            [
                prediction("scratch", scratch_mask, class_id=5),
                prediction("dent", dent_mask, class_id=2),
            ],
            [prediction("left-fender", part_mask)],
            (100, 100, 3),
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "dent")
        self.assertEqual(detections[0].damage_label, "dent")

    def test_assessment_suppresses_dent_and_scratch_when_crack_exists_on_same_part(self):
        scratch_mask = np.zeros((100, 100), dtype=np.uint8)
        scratch_mask[20:30, 20:30] = 1
        dent_mask = np.zeros((100, 100), dtype=np.uint8)
        dent_mask[35:50, 35:50] = 1
        crack_mask = np.zeros((100, 100), dtype=np.uint8)
        crack_mask[50:58, 45:55] = 1
        glass_mask = np.zeros((100, 100), dtype=np.uint8)
        glass_mask[15:25, 45:55] = 1
        part_mask = np.zeros((100, 100), dtype=np.uint8)
        part_mask[10:65, 10:65] = 1

        detections = _assessment_detections(
            [
                prediction("scratch", scratch_mask, class_id=5),
                prediction("dent", dent_mask, class_id=2),
                prediction("crack", crack_mask, class_id=1),
                prediction("glass shatter", glass_mask, class_id=3),
            ],
            [prediction("left-fender", part_mask)],
            (100, 100, 3),
        )

        self.assertEqual([detection.class_name for detection in detections], ["crack", "glass shatter"])

    def test_semantic_damage_map_extracts_connected_components(self):
        semantic_map = np.zeros((100, 100), dtype=np.uint8)
        semantic_map[20:40, 30:50] = 1
        probabilities = np.zeros((2, 100, 100), dtype=np.float32)
        probabilities[1, 20:40, 30:50] = 0.91

        predictions = _extract_damage_predictions_from_semantic_map(
            semantic_map,
            probabilities,
            {0: "background", 1: "scratch"},
            min_area=10,
            confidence_threshold=0.3,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].class_name, "scratch")
        self.assertEqual(predictions[0].box, [30.0, 20.0, 50.0, 40.0])
        self.assertAlmostEqual(predictions[0].confidence, 0.91, places=5)
        self.assertEqual(predictions[0].mask[25, 35], 1)

    def test_damage_roi_from_parts_uses_mask_extent_with_padding(self):
        part_mask = np.zeros((100, 120), dtype=np.uint8)
        part_mask[30:60, 40:80] = 1

        roi = _damage_roi_from_parts(
            [prediction("hood", part_mask)],
            (100, 120, 3),
            padding_ratio=0.10,
            min_padding=5,
        )

        self.assertEqual(roi, (35, 25, 85, 65))

    def test_damage_roi_from_parts_falls_back_to_full_image_without_parts(self):
        roi = _damage_roi_from_parts([], (100, 120, 3), padding_ratio=0.10, min_padding=5)

        self.assertEqual(roi, (0, 0, 120, 100))

    def test_project_damage_predictions_offsets_crop_coordinates(self):
        damage_mask = np.zeros((20, 30), dtype=np.uint8)
        damage_mask[3:8, 4:12] = 1
        damage = SegmentationPrediction(
            box=[4.0, 3.0, 12.0, 8.0],
            confidence=0.88,
            class_id=2,
            class_name="dent",
            mask=damage_mask,
            polygon=[[4.0, 3.0], [12.0, 3.0], [12.0, 8.0]],
        )

        projected = _project_damage_predictions_to_image([damage], (50, 20, 80, 40), (100, 120, 3))

        self.assertEqual(projected[0].box, [54.0, 23.0, 62.0, 28.0])
        self.assertEqual(projected[0].polygon, [[54.0, 23.0], [62.0, 23.0], [62.0, 28.0]])
        self.assertEqual(projected[0].mask.shape, (100, 120))
        self.assertEqual(projected[0].mask[24, 55], 1)
        self.assertEqual(projected[0].mask[4, 5], 0)


if __name__ == "__main__":
    unittest.main()
