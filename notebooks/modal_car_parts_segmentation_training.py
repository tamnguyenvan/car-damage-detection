import os
import re

import modal

GDRIVE_FILE_ID = "1fswe1oGs1GtZ_fifiQWf2OWwX_CX5v6d"
SOURCE_DATASET_DIR = "/opt/datasets/source"
DATASET_DIR = "/opt/datasets/car-parts-yolo"
SOURCE_DIR_NAMES = ("Car damages dataset", "Car parts dataset")
EXPECTED_CLASS_COUNT = 21

app = modal.App("car-parts-segmentation-training")
output_volume = modal.Volume.from_name("car-parts-segmentation-output-vol", create_if_missing=True)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_name_from_model(model_name: str) -> str:
    stem = os.path.splitext(os.path.basename(model_name))[0]
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return f"car_parts_{stem}"


def download_and_convert_dataset() -> None:
    import hashlib
    import json
    import shutil
    from pathlib import Path

    import cv2
    import numpy as np
    import yaml

    source_root = Path(SOURCE_DATASET_DIR)

    def find_source_dir():
        candidates = []
        seen = set()
        for source_dir_name in SOURCE_DIR_NAMES:
            for candidate in [source_root / source_dir_name, *source_root.rglob(source_dir_name)]:
                if candidate in seen or not (candidate / "File1" / "ann").is_dir():
                    continue
                seen.add(candidate)
                candidates.append(candidate)

        matching_candidates = []
        candidate_classes = {}
        for candidate in candidates:
            class_names = set()
            for annotation_path in (candidate / "File1" / "ann").glob("*.json"):
                with annotation_path.open() as annotation_file:
                    annotation = json.load(annotation_file)
                class_names.update(
                    obj["classTitle"] for obj in annotation.get("objects", []) if obj.get("classTitle")
                )
            candidate_classes[str(candidate)] = sorted(class_names)
            if len(class_names) == EXPECTED_CLASS_COUNT:
                matching_candidates.append(candidate)

        if len(matching_candidates) == 1:
            return matching_candidates[0]
        raise RuntimeError(
            f"Could not identify the {EXPECTED_CLASS_COUNT}-class car-parts source below {source_root}. "
            f"Candidate annotation classes: {candidate_classes}"
        )

    source_dir = find_source_dir()

    source_file_dir = source_dir / "File1"
    annotation_dir = source_file_dir / "ann"
    image_dir = source_file_dir / "img"
    metadata_path = next(
        (path for path in (source_dir / "metadata.json", source_dir / "meta.json") if path.exists()),
        None,
    )
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise RuntimeError(f"Expected ann/ and img/ directories under {source_file_dir}.")

    classes = []
    if metadata_path is not None:
        with metadata_path.open() as metadata_file:
            classes = json.load(metadata_file).get("classes", [])
    else:
        print("Metadata file not found; deriving class names from annotation classTitle values.")
    metadata_class_names = [entry["title"] for entry in classes if entry.get("shape") == "polygon"]
    observed_class_names = set()
    for annotation_path in annotation_dir.glob("*.json"):
        with annotation_path.open() as annotation_file:
            annotation = json.load(annotation_file)
        observed_class_names.update(
            obj["classTitle"] for obj in annotation.get("objects", []) if obj.get("classTitle")
        )
    if observed_class_names and set(metadata_class_names) != observed_class_names:
        print("Metadata classes disagree with annotation classTitle values; using annotation classes.")
        class_names = sorted(observed_class_names)
        classes = []
    else:
        class_names = metadata_class_names or sorted(observed_class_names)
    class_id_by_name = {name: index for index, name in enumerate(class_names)}
    class_name_by_source_id = {
        entry["id"]: entry["title"]
        for entry in classes
        if entry.get("shape") == "polygon" and entry.get("id") is not None
    }
    if not class_names:
        raise RuntimeError(f"No polygon classes found in {annotation_dir}.")
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASS_COUNT} car-part classes in {source_dir}, but found "
            f"{len(class_names)}: {class_names}. Verify that the archive contains the car-parts subset."
        )
    print(f"Using car-parts source {source_dir} with {len(class_names)} classes: {class_names}")

    output_dir = Path(DATASET_DIR)
    if (output_dir / "data.yaml").exists():
        with (output_dir / "data.yaml").open() as data_file:
            existing_names = yaml.safe_load(data_file).get("names", [])
        if isinstance(existing_names, dict):
            existing_names = [existing_names[index] for index in sorted(existing_names)]
        if existing_names == class_names:
            print(f"Reusing prepared YOLO car-parts dataset at {output_dir}.")
            return
        print(f"Discarding stale dataset at {output_dir}; its classes do not match the source metadata.")
    shutil.rmtree(output_dir, ignore_errors=True)
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def split_for(filename: str) -> str:
        bucket = int(hashlib.sha256(filename.encode()).hexdigest()[:8], 16) % 100
        return "train" if bucket < 70 else "val" if bucket < 90 else "test"

    written_images = 0
    written_segments = 0
    skipped_objects = 0
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        image_name = annotation_path.name.removesuffix(".json")
        source_image_path = image_dir / image_name
        image = cv2.imread(str(source_image_path))
        if image is None:
            print(f"Skipping annotation without readable image: {annotation_path.name}")
            continue
        height, width = image.shape[:2]
        with annotation_path.open() as annotation_file:
            annotation = json.load(annotation_file)

        split = split_for(image_name)
        label_path = output_dir / split / "labels" / f"{source_image_path.stem}.txt"
        with label_path.open("w") as label_file:
            for obj in annotation.get("objects", []):
                class_name = obj.get("classTitle") or class_name_by_source_id.get(obj.get("classId"))
                exterior = obj.get("points", {}).get("exterior", [])
                points = np.asarray(exterior, dtype=np.float32)
                if class_name not in class_id_by_name or points.ndim != 2 or len(points) < 3 or points.shape[1] != 2:
                    skipped_objects += 1
                    continue
                points[:, 0] = np.clip(points[:, 0], 0, width) / width
                points[:, 1] = np.clip(points[:, 1], 0, height) / height
                coordinates = " ".join(f"{value:.6f}" for value in points.reshape(-1))
                label_file.write(f"{class_id_by_name[class_name]} {coordinates}\n")
                written_segments += 1

        shutil.copy2(source_image_path, output_dir / split / "images" / image_name)
        written_images += 1

    if written_segments == 0:
        raise RuntimeError("No valid car-part polygons were converted; check the annotation schema.")
    if written_images < 900:
        raise RuntimeError(
            f"Expected roughly 998 car-parts images, but converted only {written_images}. "
            f"Verify source directory {source_dir}."
        )
    with (output_dir / "data.yaml").open("w") as data_file:
        yaml.safe_dump(
            {
                "path": str(output_dir),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": class_names,
            },
            data_file,
            sort_keys=False,
        )

    print(
        f"YOLO car-parts segmentation dataset prepared at {DATASET_DIR}: "
        f"{written_images} images, {written_segments} segments, {skipped_objects} objects skipped."
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "unzip")
    .pip_install("ultralytics==8.4.78", "gdown", "pyyaml", "opencv-python-headless")
    .run_commands(
        "mkdir -p /opt/datasets/source",
        f"gdown {GDRIVE_FILE_ID} -O /tmp/data.zip",
        "unzip -qq /tmp/data.zip -d /opt/datasets/source",
        "rm -f /tmp/data.zip",
    )
    .run_function(download_and_convert_dataset)
)

@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    scaledown_window=60,
    timeout=86400,
    volumes={"/runs": output_volume},
)
def train(
    model_name: str = "yolo26n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 42,
    run_name: str = "",
):
    from ultralytics import YOLO

    if "-seg" not in model_name.lower():
        raise ValueError("Car-parts training requires a segmentation checkpoint such as 'yolo26n-seg.pt'.")

    seed_everything(seed)
    if not os.path.exists(f"{DATASET_DIR}/data.yaml"):
        raise RuntimeError("Baked YOLO car-parts dataset is missing from the Modal image.")
    resolved_run_name = run_name or _run_name_from_model(model_name)
    print(f"Training {model_name} as car-parts run '{resolved_run_name}' with seed={seed}")
    model = YOLO(model_name)
    model.train(
        data=f"{DATASET_DIR}/data.yaml",
        project="/runs",
        name=resolved_run_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        seed=seed,
        deterministic=False,
        workers=8,
    )

    print("Training complete. Running validation on the best checkpoint...")
    metrics = model.val(data=f"{DATASET_DIR}/data.yaml")
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    output_volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = "yolo26n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 42,
    run_name: str = "",
):
    resolved_run_name = run_name or _run_name_from_model(model_name)
    print(f"Dispatching car-parts segmentation training job for {model_name}...")
    train.remote(model_name, epochs, imgsz, batch, seed, run_name)
    print("To retrieve the best weights after it finishes, run:")
    print(
        "modal volume get car-parts-segmentation-output-vol "
        f"/{resolved_run_name}/weights/best.pt ./models/car_parts_yolo26_seg.pt"
    )
