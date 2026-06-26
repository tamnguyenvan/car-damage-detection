# API Documentation

The service runs two YOLO26 instance-segmentation models: one segments damage categories and the other segments vehicle parts. Each damage mask is attributed to the part containing the greatest fraction of its pixels.

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

## Detection Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `box` | `number[]` | Damage bounding box as `[x1, y1, x2, y2]`. |
| `confidence` | `number` | Damage segmentation confidence. |
| `class_id` | `integer` | Damage class ID. |
| `class_name` | `string` | Damage class name. |
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
