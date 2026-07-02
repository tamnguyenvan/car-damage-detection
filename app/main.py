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
DEFAULT_DAMAGE_MODEL_PATH = "/app/models/car_damage_segformer"
DEFAULT_PARTS_MODEL_PATH = "/app/models/car_parts_yolo26_seg.pt"
DAMAGE_MIN_AREA = int(os.getenv("DAMAGE_MIN_AREA", "16"))
DAMAGE_CONFIDENCE_THRESHOLD = float(os.getenv("DAMAGE_CONFIDENCE_THRESHOLD", "0.30"))
DAMAGE_ROI_ENABLED = os.getenv("DAMAGE_ROI_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
DAMAGE_ROI_PADDING_RATIO = float(os.getenv("DAMAGE_ROI_PADDING_RATIO", "0.08"))
DAMAGE_ROI_MIN_PADDING = int(os.getenv("DAMAGE_ROI_MIN_PADDING", "32"))

DAMAGE_MODEL_PATH = os.getenv("DAMAGE_MODEL_PATH", os.getenv("MODEL_PATH", DEFAULT_DAMAGE_MODEL_PATH))
PARTS_MODEL_PATH = os.getenv("PARTS_MODEL_PATH", DEFAULT_PARTS_MODEL_PATH)
PART_COVERAGE_THRESHOLD = float(os.getenv("PART_COVERAGE_THRESHOLD", "0.50"))
INFERENCE_IMAGE_SIZE = int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))
DAMAGE_SUPPRESSION_PRIORITY = {
    "scratch": 1,
    "dent": 2,
    "crack": 3,
}
DAMAGE_PLURAL_LABELS = {
    "crack": "cracks",
    "dent": "dents",
    "glass shatter": "glass shatters",
    "lamp broken": "broken lamps",
    "scratch": "scratches",
    "tire flat": "flat tires",
}

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


@dataclass(frozen=True)
class DamageSegformerBundle:
    model: Any
    processor: Any
    device: str
    id2label: dict[int, str]


@dataclass(frozen=True)
class MatchedDamage:
    damage: SegmentationPrediction
    matched_part: Optional[SegmentationPrediction]
    part_index: Optional[int]
    coverage: Optional[float]
    iou: Optional[float]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _normalize_damage_name(class_name: str) -> str:
    return " ".join(str(class_name).replace("_", " ").replace("-", " ").strip().lower().split())


def _damage_display_label(class_name: str, count: int) -> str:
    canonical_name = _normalize_damage_name(class_name)
    if count <= 1:
        return canonical_name
    return DAMAGE_PLURAL_LABELS.get(canonical_name, f"{canonical_name}s")


def _result_display_label(damage_label: str, matched_part: Optional[SegmentationPrediction]) -> str:
    if matched_part is None:
        return damage_label
    return f"{matched_part.class_name}: {damage_label}"


def _load_parts_segmentation_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    model_task = getattr(model, "task", None)
    if model_task != "segment":
        raise ValueError(
            f"Expected a car-parts segmentation checkpoint, but loaded task={model_task!r}."
        )
    return model


def _load_damage_segformer_model(model_path: str) -> DamageSegformerBundle:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    id2label = {int(index): label for index, label in model.config.id2label.items()}
    return DamageSegformerBundle(model=model, processor=processor, device=device, id2label=id2label)


def _predict_parts_model(model: YOLO, image: np.ndarray, imgsz: int):
    return model.predict(image, imgsz=imgsz)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global damage_model, parts_model

    logger.info("Initializing SegFormer damage segmentation model from: %s", DAMAGE_MODEL_PATH)
    logger.info("Initializing car-parts segmentation model from: %s", PARTS_MODEL_PATH)
    try:
        if not os.path.exists(DAMAGE_MODEL_PATH):
            raise FileNotFoundError(f"Damage model weight file not found at: {DAMAGE_MODEL_PATH}")
        if not os.path.exists(PARTS_MODEL_PATH):
            raise FileNotFoundError(f"Car-parts model weight file not found at: {PARTS_MODEL_PATH}")

        damage_model = _load_damage_segformer_model(DAMAGE_MODEL_PATH)
        parts_model = _load_parts_segmentation_model(PARTS_MODEL_PATH)
        logger.info("SegFormer damage and YOLO car-parts segmentation models successfully loaded.")
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
        "Microservice endpoint for SegFormer car-damage semantic segmentation and "
        "YOLO car-parts instance segmentation "
        "with part-mask ROI cropping and mask-based vehicle-part attribution."
    ),
    version="3.1.0",
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


def _full_image_roi(image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    return 0, 0, width, height


def _prediction_bbox_from_mask_or_box(
    prediction: SegmentationPrediction,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> Optional[tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    if prediction.mask is not None:
        mask = _mask_for_image(prediction.mask, image_shape)
        ys, xs = np.where(mask)
        if len(xs) and len(ys):
            return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

    if not prediction.box or len(prediction.box) != 4:
        return None

    x1, y1, x2, y2 = prediction.box
    x1 = max(0, min(width, int(np.floor(x1))))
    y1 = max(0, min(height, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _damage_roi_from_parts(
    parts: list[SegmentationPrediction],
    image_shape: tuple[int, int] | tuple[int, int, int],
    padding_ratio: float = DAMAGE_ROI_PADDING_RATIO,
    min_padding: int = DAMAGE_ROI_MIN_PADDING,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    boxes = [
        bbox
        for part in parts
        if (bbox := _prediction_bbox_from_mask_or_box(part, image_shape)) is not None
    ]
    if not boxes:
        return _full_image_roi(image_shape)

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    roi_width = x2 - x1
    roi_height = y2 - y1
    padding = max(min_padding, int(round(max(roi_width, roi_height) * max(0.0, padding_ratio))))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def _project_damage_predictions_to_image(
    damages: list[SegmentationPrediction],
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[SegmentationPrediction]:
    x1, y1, x2, y2 = roi
    roi_shape = (y2 - y1, x2 - x1)
    height, width = image_shape[:2]
    projected: list[SegmentationPrediction] = []

    for damage in damages:
        full_mask = None
        if damage.mask is not None:
            roi_mask = _mask_for_image(damage.mask, roi_shape).astype(np.uint8)
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask

        box = [
            max(0.0, min(float(width), damage.box[0] + x1)),
            max(0.0, min(float(height), damage.box[1] + y1)),
            max(0.0, min(float(width), damage.box[2] + x1)),
            max(0.0, min(float(height), damage.box[3] + y1)),
        ]
        polygon = (
            [[float(x + x1), float(y + y1)] for x, y in damage.polygon]
            if damage.polygon is not None
            else None
        )
        projected.append(
            SegmentationPrediction(
                box=box,
                confidence=damage.confidence,
                class_id=damage.class_id,
                class_name=damage.class_name,
                mask=full_mask,
                polygon=polygon,
            )
        )
    return projected


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


def _polygon_from_mask(mask: np.ndarray, merge_disconnected: bool = False) -> Optional[list[list[float]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    if merge_disconnected:
        valid_contours = [contour.reshape(-1, 2) for contour in contours if len(contour) >= 3]
        if not valid_contours:
            return None
        contour_points = np.vstack(valid_contours)
        if len(contour_points) < 3:
            return None
        contour = cv2.convexHull(contour_points).reshape(-1, 2)
    else:
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


def _predict_damage_segformer(bundle: DamageSegformerBundle, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = bundle.processor(images=rgb_image, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = bundle.model(**inputs)

    logits = functional.interpolate(
        outputs.logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    semantic_map = probabilities.argmax(axis=0).astype(np.uint8)
    return semantic_map, probabilities


def _extract_damage_predictions_from_semantic_map(
    semantic_map: np.ndarray,
    probabilities: np.ndarray,
    id2label: dict[int, str],
    min_area: int = DAMAGE_MIN_AREA,
    confidence_threshold: float = DAMAGE_CONFIDENCE_THRESHOLD,
) -> list[SegmentationPrediction]:
    predictions: list[SegmentationPrediction] = []
    for class_id, class_name in sorted(id2label.items()):
        if class_id == 0:
            continue
        class_mask = (semantic_map == class_id).astype(np.uint8)
        if class_mask.sum() < min_area:
            continue

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component_mask = (labels == component_id).astype(np.uint8)
            confidence = float(probabilities[class_id][component_mask.astype(bool)].mean())
            if confidence < confidence_threshold:
                continue

            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            predictions.append(
                SegmentationPrediction(
                    box=[float(x), float(y), float(x + width), float(y + height)],
                    confidence=confidence,
                    class_id=class_id,
                    class_name=str(class_name),
                    mask=component_mask,
                    polygon=_polygon_from_mask(component_mask),
                )
            )
    return predictions


def _damage_box_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _merge_boxes(boxes: list[list[float]]) -> list[float]:
    return [
        float(min(box[0] for box in boxes)),
        float(min(box[1] for box in boxes)),
        float(max(box[2] for box in boxes)),
        float(max(box[3] for box in boxes)),
    ]


def _damage_area(damage: SegmentationPrediction) -> int:
    if damage.mask is None:
        return 0
    return int(np.asarray(damage.mask).astype(bool).sum())


def _match_damages_to_parts(
    damages: list[SegmentationPrediction],
    parts: list[SegmentationPrediction],
    image_shape: tuple[int, int, int],
) -> list[MatchedDamage]:
    part_index_by_identity = {id(part): index for index, part in enumerate(parts)}
    matched_damages = []
    for damage in damages:
        matched_part, coverage, iou = match_damage_to_part(damage.mask, parts, image_shape)
        matched_damages.append(
            MatchedDamage(
                damage=damage,
                matched_part=matched_part,
                part_index=part_index_by_identity.get(id(matched_part)) if matched_part else None,
                coverage=coverage,
                iou=iou,
            )
        )
    return matched_damages


def _suppress_lower_priority_damage(matched_damages: list[MatchedDamage]) -> list[MatchedDamage]:
    max_priority_by_part: dict[int, int] = {}
    for matched_damage in matched_damages:
        if matched_damage.part_index is None:
            continue
        damage_name = _normalize_damage_name(matched_damage.damage.class_name)
        priority = DAMAGE_SUPPRESSION_PRIORITY.get(damage_name)
        if priority is None:
            continue
        max_priority_by_part[matched_damage.part_index] = max(
            max_priority_by_part.get(matched_damage.part_index, priority),
            priority,
        )

    filtered = []
    for matched_damage in matched_damages:
        if matched_damage.part_index is None:
            filtered.append(matched_damage)
            continue
        damage_name = _normalize_damage_name(matched_damage.damage.class_name)
        priority = DAMAGE_SUPPRESSION_PRIORITY.get(damage_name)
        if priority is None or priority >= max_priority_by_part.get(matched_damage.part_index, priority):
            filtered.append(matched_damage)
    return filtered


def _group_damage_by_part_and_class(matched_damages: list[MatchedDamage]) -> list[list[MatchedDamage]]:
    grouped: dict[tuple[object, str], list[MatchedDamage]] = {}
    for index, matched_damage in enumerate(matched_damages):
        part_key: object = matched_damage.part_index if matched_damage.part_index is not None else f"unmatched-{index}"
        damage_name = _normalize_damage_name(matched_damage.damage.class_name)
        grouped.setdefault((part_key, damage_name), []).append(matched_damage)
    return list(grouped.values())


def _build_detection_from_group(
    group: list[MatchedDamage],
    image_shape: tuple[int, int, int],
) -> DetectionResult:
    representative = group[0].damage
    matched_part = group[0].matched_part
    count = len(group)
    if count == 1:
        damage_label = _damage_display_label(representative.class_name, count)
        return DetectionResult(
            box=representative.box,
            confidence=representative.confidence,
            class_id=representative.class_id,
            class_name=representative.class_name,
            damage_label=damage_label,
            damage_count=count,
            display_label=_result_display_label(damage_label, matched_part),
            damage_polygon=representative.polygon,
            car_part=matched_part.class_name if matched_part else None,
            part_confidence=matched_part.confidence if matched_part else None,
            part_coverage=group[0].coverage,
            part_iou=group[0].iou,
            car_part_polygon=matched_part.polygon if matched_part else None,
        )

    masks = [
        _mask_for_image(matched_damage.damage.mask, image_shape)
        for matched_damage in group
        if matched_damage.damage.mask is not None
    ]
    merged_mask = np.logical_or.reduce(masks).astype(np.uint8) if masks else None

    if merged_mask is not None and merged_mask.any():
        box = _damage_box_from_mask(merged_mask)
        damage_polygon = _polygon_from_mask(merged_mask, merge_disconnected=len(group) > 1)
    else:
        box = _merge_boxes([matched_damage.damage.box for matched_damage in group])
        damage_polygon = representative.polygon

    weights = [max(1, _damage_area(matched_damage.damage)) for matched_damage in group]
    confidence = float(
        np.average(
            [matched_damage.damage.confidence for matched_damage in group],
            weights=weights,
        )
    )

    coverage = None
    iou = None
    if matched_part is not None and merged_mask is not None:
        coverage, iou = _mask_metrics(merged_mask, matched_part.mask, image_shape) if matched_part.mask is not None else (None, None)

    damage_label = _damage_display_label(representative.class_name, count)
    return DetectionResult(
        box=box,
        confidence=confidence,
        class_id=representative.class_id,
        class_name=representative.class_name,
        damage_label=damage_label,
        damage_count=count,
        display_label=_result_display_label(damage_label, matched_part),
        damage_polygon=damage_polygon,
        car_part=matched_part.class_name if matched_part else None,
        part_confidence=matched_part.confidence if matched_part else None,
        part_coverage=coverage,
        part_iou=iou,
        car_part_polygon=matched_part.polygon if matched_part else None,
    )


def _assessment_detections(
    damages: list[SegmentationPrediction],
    parts: list[SegmentationPrediction],
    image_shape: tuple[int, int, int],
) -> list[DetectionResult]:
    matched_damages = _match_damages_to_parts(damages, parts, image_shape)
    filtered_damages = _suppress_lower_priority_damage(matched_damages)
    groups = _group_damage_by_part_and_class(filtered_damages)
    return [_build_detection_from_group(group, image_shape) for group in groups]


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

        logger.info("Starting YOLO car-parts segmentation and SegFormer damage ROI segmentation...")
        parts_results = await run_in_threadpool(_predict_parts_model, parts_model, image, INFERENCE_IMAGE_SIZE)
        parts = _extract_segmentation_predictions(parts_results[0], image.shape) if parts_results else []
        damage_roi = (
            _damage_roi_from_parts(parts, image.shape)
            if DAMAGE_ROI_ENABLED
            else _full_image_roi(image.shape)
        )
        roi_x1, roi_y1, roi_x2, roi_y2 = damage_roi
        damage_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
        if damage_image.size == 0:
            logger.warning("Computed an empty damage ROI %s; falling back to full-image SegFormer inference.", damage_roi)
            damage_roi = _full_image_roi(image.shape)
            roi_x1, roi_y1, roi_x2, roi_y2 = damage_roi
            damage_image = image

        logger.info(
            "Running SegFormer on damage ROI x=%d y=%d w=%d h=%d from %d part instance(s).",
            roi_x1,
            roi_y1,
            roi_x2 - roi_x1,
            roi_y2 - roi_y1,
            len(parts),
        )
        semantic_map, damage_probabilities = await run_in_threadpool(_predict_damage_segformer, damage_model, damage_image)
        damages = _extract_damage_predictions_from_semantic_map(
            semantic_map,
            damage_probabilities,
            damage_model.id2label,
        )
        damages = _project_damage_predictions_to_image(damages, damage_roi, image.shape)
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
