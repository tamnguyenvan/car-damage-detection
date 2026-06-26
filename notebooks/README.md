# Model Training Guide

This project trains two YOLO26 instance-segmentation models:

- `modal_car_damage_detection_training.py`: damage segmentation on the DrBimmer dataset. Default model: `yolo26n-seg.pt`.
- `modal_car_parts_segmentation_training.py`: vehicle-part segmentation. Default model: `yolo26n-seg.pt`.

Both models are required by the API. Detection-only YOLO and RT-DETR checkpoints are not compatible with this segmentation pipeline.

## Prerequisites

```bash
python3.11 -m venv venv_modal
source venv_modal/bin/activate
pip install modal
modal setup
```

No local GPU is required. Modal downloads, extracts, and converts the source archive while building each training image, then runs training on an H100. Each app mounts only its own output volume.

## Damage Segmentation Training

The image build runs `mkdir -p /opt/datasets/source`, then `gdown 1fswe1oGs1GtZ_fifiQWf2OWwX_CX5v6d -O /tmp/data.zip` and `unzip -qq /tmp/data.zip -d /opt/datasets/source`. A Modal `run_function` converts the annotated 8-class subset into YOLO segmentation labels and bakes them into the same image. The converter selects the source by the annotation `classTitle` values, not by the archive folder name. A stable filename hash creates 70%/20%/10% train/validation/test splits. Model artifacts are stored only in `car-damage-segmentation-output-vol`.

```bash
cd notebooks
modal run modal_car_damage_detection_training.py
```

Train a larger model or tune parameters:

```bash
modal run modal_car_damage_detection_training.py \
  --model-name yolo26m-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 42
```

The Modal function requests 8 CPUs and uses 8 data-loader workers. Training is non-deterministic by default to retain fast GPU kernels.

Retrieve the default run:

```bash
modal volume get car-damage-segmentation-output-vol \
  /car_damage_yolo26n_seg/weights/best.pt \
  ../models/car_damage_yolo26_seg.pt
```

## Car-Parts Segmentation Training

This app builds its own image from the same ZIP. Its Modal `run_function` selects the 21-class subset by annotation `classTitle`, converts it into YOLO segmentation labels, and bakes the result into that image. It uses the same deterministic 70%/20%/10% split policy. Model artifacts are stored only in `car-parts-segmentation-output-vol`.

```bash
cd notebooks
modal run modal_car_parts_segmentation_training.py
```

Use a larger checkpoint when needed:

```bash
modal run modal_car_parts_segmentation_training.py \
  --model-name yolo26m-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --seed 42
```

Retrieve the default run:

```bash
modal volume get car-parts-segmentation-output-vol \
  /car_parts_yolo26n_seg/weights/best.pt \
  ../models/car_parts_yolo26_seg.pt
```

## Serving After Training

From the repository root:

```bash
export DAMAGE_MODEL_PATH="./models/car_damage_yolo26_seg.pt"
export PARTS_MODEL_PATH="./models/car_parts_yolo26_seg.pt"
export PART_COVERAGE_THRESHOLD="0.50"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API selects the part with the highest damage-mask coverage and returns the standard mask IoU as `part_iou`. Coverage, rather than IoU, is the acceptance metric because damage masks are normally contained in much larger vehicle-part masks.
