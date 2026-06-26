# Car Damage Assessment API
![cover](./assets/cover.gif)

A containerized FastAPI service for automated car damage assessment. It runs two fine-tuned YOLO26 instance-segmentation models trained from the DrBimmer car-parts-and-damage polygon dataset: one for damage categories and one for vehicle parts.

## Key Features

- **Two segmentation models**: YOLO26 damage and vehicle-part instance segmentation.
- **Mask-aware assessment**: Each damage response includes its polygon, matched part polygon, containment coverage, and mask IoU.
- **Production API surface**: FastAPI, startup checkpoint validation, health checks, structured errors, and CPU/GPU Docker images.
- **Modal training**: Reproducible cloud training scripts for both segmentation models.

## Model Weights

Place both segmentation checkpoints in `models/`:

```bash
mkdir -p models
cp /path/to/car_damage_best.pt models/car_damage_yolo26_seg.pt
cp /path/to/car_parts_best.pt models/car_parts_yolo26_seg.pt
```

Runtime configuration:

```bash
export DAMAGE_MODEL_PATH="./models/car_damage_yolo26_seg.pt"
export PARTS_MODEL_PATH="./models/car_parts_yolo26_seg.pt"
export PART_COVERAGE_THRESHOLD="0.50"
```

Both checkpoints must have Ultralytics task `segment`. `MODEL_PATH` remains supported as a fallback for `DAMAGE_MODEL_PATH`.

## Run with Docker

```bash
docker build -t car-damage-assessment:1.0.0 .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name car-damage-assessment \
  car-damage-assessment:1.0.0
```

For GPU inference:

```bash
docker build -t car-damage-assessment:1.0.0-gpu -f Dockerfile.gpu .
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name car-damage-assessment-gpu \
  car-damage-assessment:1.0.0-gpu
```

## Local Development

```bash
python3.11 -m venv venv
source venv/bin/activate

# CPU
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export DAMAGE_MODEL_PATH="./models/car_damage_yolo26_seg.pt"
export PARTS_MODEL_PATH="./models/car_parts_yolo26_seg.pt"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

OpenAPI documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

## Quick Test

```bash
python test.py /path/to/image.jpg --save --window-width 1400 --window-height 900
```

The client draws damage masks in orange, matched part masks in green, and damage boxes in red.

## Training

```bash
cd notebooks

# Damage segmentation, default YOLO26n-seg
modal run modal_car_damage_detection_training.py

# Vehicle-part segmentation, default YOLO26n-seg
modal run modal_car_parts_segmentation_training.py
```

Retrieve the checkpoints:

```bash
modal volume get car-damage-segmentation-output-vol \
  /car_damage_yolo26n_seg/weights/best.pt ./models/car_damage_yolo26_seg.pt
modal volume get car-parts-segmentation-output-vol \
  /car_parts_yolo26n_seg/weights/best.pt ./models/car_parts_yolo26_seg.pt
```

See [notebooks/README.md](notebooks/README.md) for training details and [docs/API.md](docs/API.md) for the API contract.
