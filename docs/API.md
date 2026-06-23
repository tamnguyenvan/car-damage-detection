# API Documentation

The microservice provides the following endpoints for integration. You can also explore the interactive API documentation at `/docs` when the service is running locally.

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health probe to check service status and model availability. |
| `/predict` | `POST` | Primary inference endpoint. Accepts an image and returns bounding boxes. |

### 1. Health Probe
Checks the operational status of the service and confirms if the model weights are loaded correctly.

* **Endpoint**: `GET /health`
* **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "service": "car_damage_detector"
  }
  ```

### 2. Object Detection (Inference)
Accepts an image file payload and processes it to identify car damages.

* **Endpoint**: `POST /predict`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  - `file`: Image file (Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`)

* **Example Call (cURL)**:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/damaged_car.jpg;type=image/jpeg"
  ```

* **Example Response (200 OK)**:
  ```json
  {
    "success": true,
    "detections": [
      {
        "box": [124.5, 210.1, 350.2, 412.8],
        "confidence": 0.8754,
        "class_id": 1,
        "class_name": "bumper-dent"
      },
      {
        "box": [42.1, 150.3, 110.8, 230.5],
        "confidence": 0.7241,
        "class_id": 0,
        "class_name": "scratch"
      }
    ],
    "error": null
  }
  ```

---

## Error Handling

The API returns standard, structured JSON payloads in the event of failure to prevent breaking client-side integrations. All errors follow a predictable schema:

| HTTP Status | Error Type | Description |
| :--- | :--- | :--- |
| **400** | `Bad Request` | Raised when an unsupported file format is uploaded. |
| **422** | `Unprocessable Entity` | Raised if the submitted file cannot be decoded as an image. |
| **500** | `Internal Server Error` | Raised on generic, unexpected system-level errors. |
| **503** | `Service Unavailable` | Raised if model weights fail to load during startup or if the model is undefined. |

**Sample Error Response Payload (500 Internal Server Error)**:
```json
{
  "success": false,
  "detections": [],
  "error": "Internal processing error occurred while generating prediction results."
}
```
