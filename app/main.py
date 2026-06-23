import os
import logging
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO

from app.schemas import InferenceResponse, DetectionResult

MODEL_ARCH = "yolo-detect"

# Configure structured system logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(f"{MODEL_ARCH}-service")

# Global reference to keep model in-memory
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best.pt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup (loading weights) and shutdown (cleanup) processes.
    Fails startup immediately if weights are missing, preventing unstable deployments.
    """
    global model
    logger.info(f"Initializing {MODEL_ARCH} inference model from: {MODEL_PATH}...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weight file not found at: {MODEL_PATH}")
        
        model = YOLO(MODEL_PATH)
        model_task = getattr(model, "task", None)
        if model_task and model_task != "detect":
            raise ValueError(
                f"Expected an object-detection checkpoint, but loaded task='{model_task}'."
            )
            
        logger.info(f"{MODEL_ARCH} inference model successfully loaded.")
    except Exception as e:
        logger.critical(f"Inference engine startup failed: {str(e)}", exc_info=True)
        raise e
    
    yield
    
    # Release model assets
    logger.info("Cleaning up engine and shutting down microservice...")
    del model

app = FastAPI(
    title="Car Damage Detection API",
    description="Microservice endpoint for real-time YOLO object-detection inference on car damage images.",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensures HTTPExceptions follow the standard schema."""
    return JSONResponse(
        status_code=exc.status_code,
        content=InferenceResponse(success=False, detections=[], error=str(exc.detail)).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensures FastAPI validation errors follow the standard schema."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=InferenceResponse(success=False, detections=[], error="Invalid request parameters or missing file.").model_dump()
    )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    System endpoint for orchestrator health probes (e.g., Kubernetes, ECS).
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference system is not ready."
        )
    return {"status": "healthy", "service": "car_damage_detector"}

@app.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Inference endpoint to analyze image for car damage instances"
)
async def predict_damage(file: UploadFile = File(...)):
    """
    Processes uploaded images and extracts model predictions.
    """
    logger.info(f"Received inference request for file: '{file.filename}'")

    # 1. Image format validation
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        logger.warning(f"Rejected unsupported file extension: '{file_ext}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported file types: {', '.join(allowed_extensions)}"
        )

    try:
        # 2. Extract and decode payload bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode payload bytes into a valid image array.")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Provided image file is corrupted or unreadable."
            )

        # 3. Model validation check
        if model is None:
            logger.critical("Model container is undefined during runtime call.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is temporarily unavailable."
            )

        # 4. Model inference processing
        logger.info("Starting model inference...")
        # Run synchronous PyTorch inference in a threadpool to prevent blocking the event loop
        results = await run_in_threadpool(model.predict, img, imgsz=640)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names
            
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = names[cls_id]

                detections.append(
                    DetectionResult(
                        box=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    )
                )

        logger.info(f"Inference successfully executed. Detected {len(detections)} instances of damage.")
        return InferenceResponse(success=True, detections=detections)

    except Exception as e:
        # Catch unexpected infrastructure/hardware errors to keep endpoint responsive
        logger.error(f"Unexpected processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error occurred while generating prediction results."
        )
