from pydantic import BaseModel, Field
from typing import List, Optional

class DetectionResult(BaseModel):
    """
    Schema representing a single detected bounding box and metadata.
    """
    box: List[float] = Field(..., description="Coordinates of bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Model confidence score")
    class_id: int = Field(..., description="ID of predicted class")
    class_name: str = Field(..., description="Name of predicted class")
    damage_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Image-coordinate polygon of the damage segmentation mask",
    )
    car_part: Optional[str] = Field(default=None, description="Matched car part for the detected damage")
    part_confidence: Optional[float] = Field(default=None, description="Confidence score of the matched car-part segmentation")
    part_coverage: Optional[float] = Field(
        default=None,
        description="Fraction of the damage mask contained by the matched car-part mask",
    )
    part_iou: Optional[float] = Field(
        default=None,
        description="Intersection over union between the damage and matched car-part masks",
    )
    car_part_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Image-coordinate polygon of the matched car-part segmentation mask",
    )

class InferenceResponse(BaseModel):
    """
    Standard API response schema for inference endpoints.
    """
    success: bool = Field(..., description="Status indicating successful execution")
    detections: List[DetectionResult] = Field(default_factory=list, description="List of detected damages")
    error: Optional[str] = Field(default=None, description="Detailed error message if execution fails")
