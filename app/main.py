import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from app.schemas import DetectionResult, InferenceResponse

SERVICE_NAME = "car_damage_assessment"
DEFAULT_DAMAGE_MODEL_PATH = "/app/models/car_damage_yolo26_seg.pt"
DEFAULT_PARTS_MODEL_PATH = "/app/models/car_parts_yolo26_seg.pt"

DAMAGE_MODEL_PATH = os.getenv("DAMAGE_MODEL_PATH", os.getenv("MODEL_PATH", DEFAULT_DAMAGE_MODEL_PATH))
PARTS_MODEL_PATH = os.getenv("PARTS_MODEL_PATH", DEFAULT_PARTS_MODEL_PATH)
PART_COVERAGE_THRESHOLD = float(os.getenv("PART_COVERAGE_THRESHOLD", "0.50"))
INFERENCE_IMAGE_SIZE = int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(f"{SERVICE_NAME}-service")

damage_model = None
parts_model = None


@dataclass(frozen=True)
class SegmentationPrediction:
    box: list[float]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray]
    polygon: Optional[list[list[float]]]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _load_segmentation_model(model_path: str, model_label: str) -> YOLO:
    model = YOLO(model_path)
    model_task = getattr(model, "task", None)
    if model_task != "segment":
        raise ValueError(
            f"Expected a {model_label} segmentation checkpoint, but loaded task={model_task!r}."
        )
    return model


def _predict_model(model: YOLO, image: np.ndarray, imgsz: int):
    return model.predict(image, imgsz=imgsz)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global damage_model, parts_model

    logger.info("Initializing damage segmentation model from: %s", DAMAGE_MODEL_PATH)
    logger.info("Initializing car-parts segmentation model from: %s", PARTS_MODEL_PATH)
    try:
        if not os.path.exists(DAMAGE_MODEL_PATH):
            raise FileNotFoundError(f"Damage model weight file not found at: {DAMAGE_MODEL_PATH}")
        if not os.path.exists(PARTS_MODEL_PATH):
            raise FileNotFoundError(f"Car-parts model weight file not found at: {PARTS_MODEL_PATH}")

        damage_model = _load_segmentation_model(DAMAGE_MODEL_PATH, "damage")
        parts_model = _load_segmentation_model(PARTS_MODEL_PATH, "car-parts")
        logger.info("Damage and car-parts segmentation models successfully loaded.")
    except Exception as exc:
        logger.critical("Inference engine startup failed: %s", exc, exc_info=True)
        raise

    yield

    logger.info("Cleaning up inference models and shutting down microservice...")
    damage_model = None
    parts_model = None


app = FastAPI(
    title="Car Damage Assessment API",
    description=(
        "Microservice endpoint for car-damage and car-parts instance segmentation "
        "with mask-based vehicle-part attribution."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=InferenceResponse(success=False, detections=[], error=str(exc.detail)).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=InferenceResponse(
            success=False,
            detections=[],
            error="Invalid request parameters or missing file.",
        ).model_dump(),
    )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if damage_model is None or parts_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference system is not ready.",
        )
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "models": {
            "damage_segmentation_model": "ready",
            "parts_segmentation_model": "ready",
        },
    }


def _mask_for_image(mask: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    binary_mask = np.asarray(mask).astype(bool)
    if binary_mask.shape[:2] != (height, width):
        binary_mask = cv2.resize(binary_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return binary_mask


def _mask_metrics(
    damage_mask: np.ndarray,
    part_mask: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[float, float]:
    damage = _mask_for_image(damage_mask, image_shape)
    part = _mask_for_image(part_mask, image_shape)
    intersection = np.logical_and(damage, part).sum()
    damage_area = damage.sum()
    union = np.logical_or(damage, part).sum()
    if damage_area == 0 or union == 0:
        return 0.0, 0.0
    return float(intersection / damage_area), float(intersection / union)


def match_damage_to_part(
    damage_mask: Optional[np.ndarray],
    parts: list[SegmentationPrediction],
    image_shape: tuple[int, int] | tuple[int, int, int],
    threshold: float = PART_COVERAGE_THRESHOLD,
) -> tuple[Optional[SegmentationPrediction], Optional[float], Optional[float]]:
    """Match by damage-mask containment and expose symmetric IoU for review."""
    if damage_mask is None:
        return None, None, None

    best_part = None
    best_coverage = 0.0
    best_iou = 0.0
    for part in parts:
        if part.mask is None:
            continue
        coverage, iou = _mask_metrics(damage_mask, part.mask, image_shape)
        if coverage > best_coverage or (coverage == best_coverage and iou > best_iou):
            best_part = part
            best_coverage = coverage
            best_iou = iou

    if best_part is None or best_coverage < threshold:
        return None, None, None
    return best_part, best_coverage, best_iou


def _polygon_from_mask(mask: np.ndarray) -> Optional[list[list[float]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        return None
    return [[float(x), float(y)] for x, y in contour.tolist()]


def _segmentation_geometry_from_result(
    result: Any,
    count: int,
    image_shape: tuple[int, int, int],
) -> tuple[list[Optional[np.ndarray]], list[Optional[list[list[float]]]]]:
    if count == 0 or getattr(result, "masks", None) is None:
        return [None] * count, [None] * count

    height, width = image_shape[:2]
    masks = result.masks
    polygons = getattr(masks, "xy", None)
    if polygons is not None:
        output_masks: list[Optional[np.ndarray]] = []
        output_polygons: list[Optional[list[list[float]]]] = []
        for polygon in polygons[:count]:
            points = np.asarray(polygon, dtype=np.float32)
            if points.ndim != 2 or len(points) < 3 or points.shape[1] != 2:
                output_masks.append(None)
                output_polygons.append(None)
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
            output_masks.append(mask)
            output_polygons.append([[float(x), float(y)] for x, y in points.tolist()])
        output_masks.extend([None] * (count - len(output_masks)))
        output_polygons.extend([None] * (count - len(output_polygons)))
        return output_masks, output_polygons

    mask_data = getattr(masks, "data", None)
    if mask_data is None:
        return [None] * count, [None] * count

    mask_array = mask_data.detach().cpu().numpy() if hasattr(mask_data, "detach") else np.asarray(mask_data)
    output_masks: list[Optional[np.ndarray]] = []
    output_polygons: list[Optional[list[list[float]]]] = []
    for raw_mask in mask_array[:count]:
        mask = _mask_for_image(raw_mask > 0.5, image_shape).astype(np.uint8)
        output_masks.append(mask)
        output_polygons.append(_polygon_from_mask(mask))
    output_masks.extend([None] * (count - len(output_masks)))
    output_polygons.extend([None] * (count - len(output_polygons)))
    return output_masks, output_polygons


def _extract_segmentation_predictions(
    result: Any,
    image_shape: tuple[int, int, int],
) -> list[SegmentationPrediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    count = len(boxes)
    names = getattr(result, "names", {})
    masks, polygons = _segmentation_geometry_from_result(result, count, image_shape)
    predictions = []
    for index, box in enumerate(boxes):
        predictions.append(
            SegmentationPrediction(
                box=[float(value) for value in box.xyxy[0].tolist()],
                confidence=float(box.conf[0].item()),
                class_id=int(box.cls[0].item()),
                class_name=_class_name(names, int(box.cls[0].item())),
                mask=masks[index],
                polygon=polygons[index],
            )
        )
    return predictions


def _assessment_detections(
    damages: list[SegmentationPrediction],
    parts: list[SegmentationPrediction],
    image_shape: tuple[int, int, int],
) -> list[DetectionResult]:
    detections = []
    for damage in damages:
        matched_part, coverage, iou = match_damage_to_part(damage.mask, parts, image_shape)
        detections.append(
            DetectionResult(
                box=damage.box,
                confidence=damage.confidence,
                class_id=damage.class_id,
                class_name=damage.class_name,
                damage_polygon=damage.polygon,
                car_part=matched_part.class_name if matched_part else None,
                part_confidence=matched_part.confidence if matched_part else None,
                part_coverage=coverage,
                part_iou=iou,
                car_part_polygon=matched_part.polygon if matched_part else None,
            )
        )
    return detections


@app.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment car damage and attribute each damage mask to a vehicle part",
)
async def predict_damage(file: UploadFile = File(...)):
    logger.info("Received inference request for file: '%s'", file.filename)

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported file types: {', '.join(sorted(allowed_extensions))}",
        )

    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Provided image file is corrupted or unreadable.",
            )
        if damage_model is None or parts_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is temporarily unavailable.",
            )

        logger.info("Starting damage and car-parts segmentation...")
        damage_results = await run_in_threadpool(_predict_model, damage_model, image, INFERENCE_IMAGE_SIZE)
        parts_results = await run_in_threadpool(_predict_model, parts_model, image, INFERENCE_IMAGE_SIZE)
        damages = _extract_segmentation_predictions(damage_results[0], image.shape) if damage_results else []
        parts = _extract_segmentation_predictions(parts_results[0], image.shape) if parts_results else []
        detections = _assessment_detections(damages, parts, image.shape)

        logger.info("Inference executed. Segmented %d damage instance(s).", len(detections))
        return InferenceResponse(success=True, detections=detections)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected processing error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error occurred while generating prediction results.",
        )
