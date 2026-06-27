# Model Training Guide

This project trains two segmentation models:

- `modal_car_damage_detection_training.py`: SegFormer semantic damage segmentation on CarDD. Default model: `nvidia/mit-b2`.
- `modal_car_parts_segmentation_training.py`: vehicle-part segmentation. Default model: `yolo26n-seg.pt`.

Both models are required by the API. SegFormer produces semantic damage masks that the API splits into connected damage regions; YOLO26 still provides vehicle-part instance masks.

## Prerequisites

```bash
python3.11 -m venv venv_modal
source venv_modal/bin/activate
pip install modal
modal setup
```

No local GPU is required. Modal downloads, extracts, and converts the source archives while building each training image, then runs training on an H100. Each app mounts only its own output volume.

## Damage Segmentation Training

The image build installs CUDA PyTorch, Torchvision, Transformers, and `wget`, downloads `https://huggingface.co/datasets/tamnvcc/CarDD/resolve/main/CarDD_release.zip`, extracts it, and runs a Modal `run_function` to convert COCO polygon/RLE masks into semantic segmentation masks. Background is class `0`; CarDD damage classes start at class `1`. Model artifacts are stored only in `car-damage-segformer-output-vol`.

```bash
cd notebooks
modal run modal_car_damage_detection_training.py
```

Train a larger model or tune parameters:

```bash
modal run modal_car_damage_detection_training.py \
  --model-name nvidia/mit-b3 \
  --epochs 50 \
  --image-size 512 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 0.00006 \
  --seed 42
```

The Modal function requests 8 CPUs and uses 8 Trainer data-loader workers. It evaluates at the end of every epoch, logs epoch-level metrics with tqdm disabled for readable Modal logs, and saves the best checkpoint by `mean_iou_without_background`. It also reports mean Dice, mean precision, mean recall, mean F1, damage false-positive rate, damage false-negative rate, and per-class IoU/Dice. After training, it saves `training_summary.png` with loss, quality metrics, precision/recall, error rates, learning rate, and per-class bars from the best epoch. The final Hugging Face model directory is saved under each run as `model/`.

Retrieve the default run:

```bash
modal volume get car-damage-segformer-output-vol \
  /car_damage_segformer_mit_b2/model \
  ../models/car_damage_segformer
modal volume get car-damage-segformer-output-vol \
  /car_damage_segformer_mit_b2/training_summary.png \
  ./car_damage_segformer_training_summary.png
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
export DAMAGE_MODEL_PATH="./models/car_damage_segformer"
export PARTS_MODEL_PATH="./models/car_parts_yolo26_seg.pt"
export DAMAGE_CONFIDENCE_THRESHOLD="0.30"
export DAMAGE_MIN_AREA="16"
export DAMAGE_ROI_ENABLED="true"
export DAMAGE_ROI_PADDING_RATIO="0.08"
export DAMAGE_ROI_MIN_PADDING="32"
export PART_COVERAGE_THRESHOLD="0.50"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API runs car-parts segmentation first, crops a padded vehicle ROI for SegFormer damage segmentation, maps damage masks back to the original image, then selects the part with the highest damage-mask coverage. It returns the standard mask IoU as `part_iou`. Coverage, rather than IoU, is the acceptance metric because damage masks are normally contained in much larger vehicle-part masks.
