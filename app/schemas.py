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

class InferenceResponse(BaseModel):
    """
    Standard API response schema for inference endpoints.
    """
    success: bool = Field(..., description="Status indicating successful execution")
    detections: List[DetectionResult] = Field(default=[], description="List of detected damages")
    error: Optional[str] = Field(default=None, description="Detailed error message if execution fails")