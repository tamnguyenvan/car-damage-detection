# API Documentation

The service runs a Hugging Face SegFormer semantic-segmentation model for damage categories and a YOLO26 instance-segmentation model for vehicle parts. SegFormer damage maps are split into connected regions, each damage region is attributed to the part containing the greatest fraction of its pixels, and final results are grouped by vehicle part and damage type.

## Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Checks readiness of both segmentation models. |
| `/predict` | `POST` | Segments damage and returns its matched vehicle part. |

## Health Probe

**Endpoint**: `GET /health`

```json
{
  "status": "healthy",
  "service": "car_damage_assessment",
  "models": {
    "damage_segmentation_model": "ready",
    "parts_segmentation_model": "ready"
  }
}
```

## Prediction

**Endpoint**: `POST /predict`
**Content-Type**: `multipart/form-data`

- `file`: image file with extension `.jpg`, `.jpeg`, `.png`, or `.webp`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/damaged_car.jpg;type=image/jpeg"
```

**Example response (200 OK)**:

```json
{
  "success": true,
  "detections": [
    {
      "box": [124.5, 210.1, 350.2, 412.8],
      "confidence": 0.8754,
      "class_id": 1,
      "class_name": "dent",
      "damage_label": "dent",
      "damage_count": 1,
      "display_label": "front_bumper: dent",
      "damage_polygon": [[145.0, 255.0], [190.0, 245.0], [215.0, 290.0]],
      "car_part": "front_bumper",
      "part_confidence": 0.9123,
      "part_coverage": 0.9431,
      "part_iou": 0.0417,
      "car_part_polygon": [[124.0, 230.0], [348.0, 230.0], [350.0, 410.0]]
    }
  ],
  "error": null
}
```

## Matching Semantics

`part_coverage` is the fraction of damage-mask pixels inside the selected part mask: `intersection(damage, part) / area(damage)`. It is used for matching because vehicle-part masks are normally much larger than the contained damage mask.

`part_iou` is also returned as the standard symmetric mask IoU: `intersection(damage, part) / union(damage, part)`. It is useful for auditing overlap, but is not used as the acceptance threshold because small damages would produce very low IoU values even when correctly matched.

Set `PART_COVERAGE_THRESHOLD` to control the minimum coverage needed for a matched part. The default is `0.50`.

Set `DAMAGE_CONFIDENCE_THRESHOLD` to filter low-confidence SegFormer damage regions and `DAMAGE_MIN_AREA` to drop tiny connected components.

By default, the API runs car-parts segmentation first, builds a padded ROI around the detected part masks, runs SegFormer on that crop, and maps damage masks back to original image coordinates. This preserves more detail for full-scene photos than resizing the whole image into SegFormer's input size. Set `DAMAGE_ROI_ENABLED=false` to disable the crop. Tune the crop with `DAMAGE_ROI_PADDING_RATIO` and `DAMAGE_ROI_MIN_PADDING`.

## Result Grouping

Same-class damage regions on the same matched vehicle part are merged into one result. For example, multiple scratch components on `left-fender` become one result with `class_name: "scratch"`, `damage_label: "scratches"`, `damage_count` set to the number of merged scratch regions, and `display_label: "left-fender: scratches"`. The returned `box` spans the merged damage mask.

Within the same vehicle part, lower-priority surface damage is suppressed when a stronger related damage exists: `dent` suppresses `scratch`, and `crack` suppresses both `dent` and `scratch`. Other damage classes, such as `glass shatter`, `lamp broken`, and `tire flat`, are still reported independently.

## Detection Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `box` | `number[]` | Damage bounding box as `[x1, y1, x2, y2]`. |
| `confidence` | `number` | Mean SegFormer class probability over the damage region. |
| `class_id` | `integer` | Damage class ID. |
| `class_name` | `string` | Canonical singular damage class name. |
| `damage_label` | `string | null` | Display label, pluralized when multiple same-class regions are merged. |
| `damage_count` | `integer` | Number of same-class regions merged into this result. |
| `display_label` | `string | null` | Human-readable part and damage label for overlays. |
| `damage_polygon` | `number[][] | null` | Image-coordinate polygon of the damage mask. |
| `car_part` | `string | null` | Matched vehicle-part class name. |
| `part_confidence` | `number | null` | Matched part-segmentation confidence. |
| `part_coverage` | `number | null` | Fraction of damage mask contained by the selected part mask. |
| `part_iou` | `number | null` | Symmetric mask IoU for the damage and selected part. |
| `car_part_polygon` | `number[][] | null` | Image-coordinate polygon of the matched part mask. |

## Errors

All errors use the response shape below:

```json
{
  "success": false,
  "detections": [],
  "error": "Internal processing error occurred while generating prediction results."
}
```

| HTTP Status | Description |
| :--- | :--- |
| `400` | Unsupported file extension. |
| `422` | Missing file or unreadable image payload. |
| `500` | Unexpected processing error. |
| `503` | One or both segmentation models are unavailable. |
