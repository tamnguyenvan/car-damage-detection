import os
import modal

# Define the Modal App
app = modal.App("car-damage-detector-training")

# Define the Volume to persist training results
volume = modal.Volume.from_name("car-damage-training-vol", create_if_missing=True)


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def download_and_convert_dataset():
    """
    This function runs during the Modal Image build phase.
    It downloads the COCO dataset, extracts it, converts it to YOLO detection format,
    and cleans up the original zip to keep the image lightweight.
    """
    import json
    import shutil
    import yaml
    import subprocess

    # 1. Download dataset using gdown
    print("Downloading CarDD_release.zip...")
    subprocess.run(["gdown", "1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx"], check=True)

    # 2. Extract dataset
    print("Extracting dataset...")
    subprocess.run(["unzip", "-qq", "CarDD_release.zip"], check=True)

    # 3. Define Paths
    coco_root_dir = "CarDD_release/CarDD_COCO"
    yolo_output_dir = "/CarDD"

    if os.path.exists(yolo_output_dir):
        shutil.rmtree(yolo_output_dir)
    os.makedirs(yolo_output_dir, exist_ok=True)

    def coco_to_yolo_bbox(bbox, img_width, img_height):
        x_min, y_min, bbox_width, bbox_height = bbox
        x_center = (x_min + bbox_width / 2) / img_width
        y_center = (y_min + bbox_height / 2) / img_height
        norm_bbox_width = bbox_width / img_width
        norm_bbox_height = bbox_height / img_height
        return [
            min(max(x_center, 0.0), 1.0),
            min(max(y_center, 0.0), 1.0),
            min(max(norm_bbox_width, 0.0), 1.0),
            min(max(norm_bbox_height, 0.0), 1.0),
        ]

    split_map = {
        'train': {'image_dir': 'train2017', 'json_file': 'instances_train2017.json'},
        'val': {'image_dir': 'val2017', 'json_file': 'instances_val2017.json'},
        'test': {'image_dir': 'test2017', 'json_file': 'instances_test2017.json'},
    }

    all_class_names = {}

    for split_name, paths in split_map.items():
        coco_json_path = os.path.join(coco_root_dir, "annotations", paths['json_file'])
        try:
            with open(coco_json_path, 'r') as f:
                coco_data = json.load(f)
        except Exception:
            continue
        for cat in coco_data['categories']:
            all_class_names[cat['id']] = cat['name']

    class_names = [all_class_names[cat_id] for cat_id in sorted(all_class_names.keys())]
    class_name_to_yolo_id = {name: i for i, name in enumerate(class_names)}

    for split_name, paths in split_map.items():
        split_images_dir_source = os.path.join(coco_root_dir, paths['image_dir'])
        coco_json_path = os.path.join(coco_root_dir, "annotations", paths['json_file'])

        if not os.path.exists(split_images_dir_source) or not os.path.exists(coco_json_path):
            continue

        images_output_dir = os.path.join(yolo_output_dir, split_name, 'images')
        labels_output_dir = os.path.join(yolo_output_dir, split_name, 'labels')
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        images_info = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {img['id']: [] for img in coco_data['images']}
        for ann in coco_data['annotations']:
            bbox = ann.get('bbox')
            if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            annotations_by_image[ann['image_id']].append(ann)

        for img_id, img_info in images_info.items():
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            src_image_path = os.path.join(split_images_dir_source, img_filename)
            dst_image_path = os.path.join(images_output_dir, img_filename)

            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                continue

            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_file_path = os.path.join(labels_output_dir, label_filename)

            with open(label_file_path, 'w') as f_label:
                for ann in annotations_by_image.get(img_id, []):
                    coco_category_id = ann['category_id']
                    category_name = all_class_names.get(coco_category_id)
                    if category_name:
                        yolo_class_id = class_name_to_yolo_id[category_name]
                        yolo_bbox = coco_to_yolo_bbox(ann['bbox'], img_width, img_height)
                        f_label.write(
                            f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n"
                        )

    # Generate data.yaml for YOLO detection
    data_yaml_path = os.path.join(yolo_output_dir, 'data.yaml')
    data_yaml_content = {
        'path': os.path.abspath(yolo_output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    print("Cleanup zip and extracted files...")
    os.remove("CarDD_release.zip")
    shutil.rmtree("CarDD_release")
    print("YOLO detection dataset baked successfully into /CarDD")


# Define the Image
# 1. Base Debian Python 3.10 image
# 2. apt-get install necessary system libraries for OpenCV
# 3. pip install required Python packages
# 4. run the download_and_convert_dataset function during image build
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "wget", "unzip")
    .pip_install(
        "ultralytics",
        "gdown",
        "pyyaml",
        "opencv-python-headless",
    )
    .run_function(download_and_convert_dataset)
)


@app.function(
    image=image,
    gpu="H100",  # or A100, L40S
    scaledown_window=60,
    timeout=86400, # 24 hours max
    volumes={"/runs": volume}
)
def train(model_name: str = "yolov8m.pt", seed: int = 42):
    from ultralytics import YOLO

    seed_everything(seed)
    print(f"Seeded Python, NumPy, PyTorch, and CUDA with seed={seed}")

    print(f"Initializing {model_name} model...")
    if "-seg" in model_name.lower():
        raise ValueError(
            "Object detection training requires a detection model, "
            "for example 'yolov8n.pt', 'yolov8m.pt', or 'yolo11m.pt'."
        )
    model = YOLO(model_name)

    print("Starting training...")
    # Train the model, saving outputs directly to the mounted volume
    model.train(
        data="/CarDD/data.yaml",
        project="/runs",
        name="car_damage_finetuned",
        epochs=100,
        imgsz=640,
        device=0,
        seed=seed,
        deterministic=True,
        workers=0,
    )

    print("Training complete! Model artifacts are saved in the Modal Volume.")

    # Validation
    metrics = model.val()

    print("\n--- Validation Performance Metrics ---")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


@app.local_entrypoint()
def main(model_name: str = "yolov8m.pt", seed: int = 42):
    print(f"Dispatching training job for {model_name} to Modal...")
    train.remote(model_name, seed)
    print("\nTraining job submitted!")
    print("To retrieve your model weights after it finishes, run:")
    print(
        "modal volume get car-damage-training-vol "
        "/car_damage_finetuned/weights/best.pt ./models/best.pt"
    )
