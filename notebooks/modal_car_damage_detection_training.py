import os
import re
import shutil
from pathlib import Path

import modal

GDRIVE_ID = "1YCdgmku7Wad1SB3VDKpY90_pxD5QCubX"
SOURCE_ROOT = "/opt/datasets/source"
CARDD_COCO_DIR = "/opt/datasets/source/CarDD_release/CarDD_COCO"
DATASET_DIR = "/opt/datasets/cardd-segformer"
DAMAGE_CLASSES = ["scratch", "dent", "crack", "glass shatter", "lamp broken", "tire flat"]
BACKGROUND_LABEL = "background"

app = modal.App("car-damage-segformer-training")
output_volume = modal.Volume.from_name("car-damage-segformer-output-vol", create_if_missing=True)


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
    stem = os.path.basename(model_name.rstrip("/"))
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return f"car_damage_segformer_{stem}"


def download_and_convert_dataset() -> None:
    import json

    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    source_dir = Path(CARDD_COCO_DIR)
    if not source_dir.is_dir():
        raise RuntimeError(f"Expected CarDD COCO dataset at {source_dir}.")

    output_dir = Path(DATASET_DIR)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and (output_dir / "train" / "masks").is_dir():
        print(f"Reusing prepared SegFormer CarDD dataset at {output_dir}.")
        return

    shutil.rmtree(output_dir, ignore_errors=True)
    id2label = {0: BACKGROUND_LABEL}
    id2label.update({index + 1: class_name for index, class_name in enumerate(DAMAGE_CLASSES)})
    label2id = {label: index for index, label in id2label.items()}
    canonical_label_by_name = {
        label.replace("_", " ").replace("-", " ").lower(): index
        for label, index in label2id.items()
    }

    split_map = {
        "train": {"image_dir": "train2017", "json_file": "instances_train2017.json"},
        "val": {"image_dir": "val2017", "json_file": "instances_val2017.json"},
        "test": {"image_dir": "test2017", "json_file": "instances_test2017.json"},
    }

    def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if isinstance(segmentation, list):
            polygons = []
            for polygon in segmentation:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 3:
                    polygons.append(np.rint(points).astype(np.int32))
            if polygons:
                cv2.fillPoly(mask, polygons, 1)
            return mask

        if isinstance(segmentation, dict):
            decoded = mask_utils.decode(segmentation)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            return decoded.astype(np.uint8)

        return mask

    total_images = 0
    total_segments = 0
    for split_name, split_paths in split_map.items():
        image_source_dir = source_dir / split_paths["image_dir"]
        annotation_path = source_dir / "annotations" / split_paths["json_file"]
        if not image_source_dir.is_dir() or not annotation_path.exists():
            raise RuntimeError(f"Missing CarDD {split_name} files below {source_dir}.")

        image_output_dir = output_dir / split_name / "images"
        mask_output_dir = output_dir / split_name / "masks"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        with annotation_path.open() as annotation_file:
            coco_data = json.load(annotation_file)

        category_id_to_name = {category["id"]: category["name"] for category in coco_data["categories"]}
        category_id_to_label = {
            category_id: canonical_label_by_name[name.replace("_", " ").replace("-", " ").lower()]
            for category_id, name in category_id_to_name.items()
            if name.replace("_", " ").replace("-", " ").lower() in canonical_label_by_name
        }
        unknown_classes = sorted(
            name
            for name in set(category_id_to_name.values())
            if name.replace("_", " ").replace("-", " ").lower() not in canonical_label_by_name
        )
        if unknown_classes:
            raise RuntimeError(f"Unexpected CarDD classes in {annotation_path}: {unknown_classes}")

        annotations_by_image = {image["id"]: [] for image in coco_data["images"]}
        for annotation in coco_data["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image_info in coco_data["images"]:
            image_name = image_info["file_name"]
            image_path = image_source_dir / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            height, width = image.shape[:2]
            semantic_mask = np.zeros((height, width), dtype=np.uint8)
            for annotation in annotations_by_image.get(image_info["id"], []):
                class_id = category_id_to_label.get(annotation["category_id"])
                if class_id is None:
                    continue
                instance_mask = decode_segmentation(annotation.get("segmentation", []), height, width)
                if instance_mask.any():
                    semantic_mask[instance_mask.astype(bool)] = class_id
                    total_segments += 1

            shutil.copy2(image_path, image_output_dir / image_name)
            cv2.imwrite(str(mask_output_dir / f"{Path(image_name).stem}.png"), semantic_mask)
            total_images += 1

    if total_images < 3000:
        raise RuntimeError(f"Expected the full CarDD dataset, but converted only {total_images} images.")
    if total_segments == 0:
        raise RuntimeError("No CarDD damage segmentation masks were converted.")

    metadata = {
        "id2label": {str(index): label for index, label in id2label.items()},
        "label2id": label2id,
        "classes": [id2label[index] for index in sorted(id2label)],
    }
    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(
        f"SegFormer CarDD dataset prepared at {DATASET_DIR}: "
        f"{total_images} images, {total_segments} segments."
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "unzip", "wget")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate",
        "albumentations",
        "matplotlib",
        "opencv-python-headless",
        "pillow",
        "pycocotools",
        "transformers[torch]",
        "gdown"
    )
    .run_commands(
        f"mkdir -p {SOURCE_ROOT}",
        f"gdown {GDRIVE_ID} -O {SOURCE_ROOT}/CarDD_release.zip",
        f"unzip -qq {SOURCE_ROOT}/CarDD_release.zip -d {SOURCE_ROOT}",
        f"rm -f {SOURCE_ROOT}/CarDD_release.zip",
    )
    .run_function(download_and_convert_dataset)
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    scaledown_window=60,
    timeout=86400,
    volumes={"/runs": output_volume},
)
def train(
    model_name: str = "nvidia/mit-b2",
    epochs: int = 50,
    image_size: int = 512,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 6e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    run_name: str = "",
):
    import json

    import numpy as np
    import torch
    from PIL import Image
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
        Trainer,
        TrainingArguments,
    )

    seed_everything(seed)
    dataset_root = Path(DATASET_DIR)
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Baked CarDD SegFormer dataset metadata is missing from the Modal image.")

    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    id2label = {int(index): label for index, label in metadata["id2label"].items()}
    label2id = {label: int(index) for label, index in metadata["label2id"].items()}

    class CarDDSegmentationDataset(torch.utils.data.Dataset):
        def __init__(self, split: str, processor):
            image_dir = dataset_root / split / "images"
            mask_dir = dataset_root / split / "masks"
            self.samples = [
                (image_path, mask_dir / f"{image_path.stem}.png")
                for image_path in sorted(image_dir.iterdir())
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if not self.samples:
                raise RuntimeError(f"No SegFormer samples found for split {split}.")
            self.processor = processor

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index: int):
            image_path, mask_path = self.samples[index]
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)
            encoded = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
            return {
                "pixel_values": encoded["pixel_values"].squeeze(0),
                "labels": encoded["labels"].squeeze(0).long(),
            }

    processor = SegformerImageProcessor(
        do_reduce_labels=False,
        do_resize=True,
        size={"height": image_size, "width": image_size},
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = CarDDSegmentationDataset("train", processor)
    eval_dataset = CarDDSegmentationDataset("val", processor)

    def preprocess_logits_for_metrics(logits, labels):
        import torch.nn.functional as functional

        if isinstance(logits, tuple):
            logits = logits[0]
        logits = functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits.argmax(dim=1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if predictions.ndim == 4:
            predictions = np.argmax(predictions, axis=1)
        labels = labels.astype(np.int64)
        valid_mask = labels != 255

        pixel_accuracy = float((predictions[valid_mask] == labels[valid_mask]).mean()) if valid_mask.any() else 0.0
        ious = []
        dices = []
        precisions = []
        recalls = []
        f1_scores = []
        metrics = {"pixel_accuracy": pixel_accuracy}
        for class_id in range(1, len(id2label)):
            metric_label = id2label[class_id].replace(" ", "_").replace("-", "_")
            predicted = (predictions == class_id) & valid_mask
            target = (labels == class_id) & valid_mask
            true_positive = np.logical_and(predicted, target).sum()
            false_positive = np.logical_and(predicted, ~target & valid_mask).sum()
            false_negative = np.logical_and(~predicted & valid_mask, target).sum()
            union = np.logical_or(predicted, target).sum()
            if union == 0:
                continue
            iou = float(true_positive / union)
            dice = float((2 * true_positive) / ((2 * true_positive) + false_positive + false_negative))
            precision = float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0
            recall = float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0
            f1 = float((2 * precision * recall) / (precision + recall)) if precision + recall else 0.0
            metrics[f"iou_{metric_label}"] = iou
            metrics[f"dice_{metric_label}"] = dice
            ious.append(iou)
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        predicted_damage = (predictions > 0) & valid_mask
        target_damage = (labels > 0) & valid_mask
        false_positive_damage = np.logical_and(predicted_damage, ~target_damage).sum()
        false_negative_damage = np.logical_and(~predicted_damage & valid_mask, target_damage).sum()
        true_negative_damage = np.logical_and(~predicted_damage & valid_mask, ~target_damage).sum()
        true_positive_damage = np.logical_and(predicted_damage, target_damage).sum()

        metrics.update(
            {
                "mean_iou_without_background": float(np.mean(ious)) if ious else 0.0,
                "mean_dice_without_background": float(np.mean(dices)) if dices else 0.0,
                "mean_precision_without_background": float(np.mean(precisions)) if precisions else 0.0,
                "mean_recall_without_background": float(np.mean(recalls)) if recalls else 0.0,
                "mean_f1_without_background": float(np.mean(f1_scores)) if f1_scores else 0.0,
                "damage_false_positive_rate": (
                    float(false_positive_damage / (false_positive_damage + true_negative_damage))
                    if false_positive_damage + true_negative_damage
                    else 0.0
                ),
                "damage_false_negative_rate": (
                    float(false_negative_damage / (false_negative_damage + true_positive_damage))
                    if false_negative_damage + true_positive_damage
                    else 0.0
                ),
                "predicted_damage_pixel_ratio": float(predicted_damage.sum() / valid_mask.sum()) if valid_mask.any() else 0.0,
            }
        )
        return metrics

    def save_training_plots(log_history: list[dict], output_path: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def series(metric: str, eval_metric: bool = False):
            key = f"eval_{metric}" if eval_metric else metric
            rows = [entry for entry in log_history if key in entry and "epoch" in entry]
            return [float(entry["epoch"]) for entry in rows], [float(entry[key]) for entry in rows]

        eval_rows = [entry for entry in log_history if "eval_mean_iou_without_background" in entry]
        best_eval = max(
            eval_rows,
            key=lambda entry: float(entry.get("eval_mean_iou_without_background", -1.0)),
            default={},
        )

        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        axes = axes.reshape(-1)
        fig.suptitle("CarDD SegFormer Training Summary", fontsize=16)

        epochs, values = series("loss")
        if values:
            axes[0].plot(epochs, values, marker="o", label="train loss")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        for metric, label in (
            ("mean_iou_without_background", "mIoU"),
            ("mean_dice_without_background", "Dice"),
            ("mean_f1_without_background", "F1"),
        ):
            epochs, values = series(metric, eval_metric=True)
            if values:
                axes[1].plot(epochs, values, marker="o", label=label)
        axes[1].set_title("Segmentation Quality")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        for metric, label in (
            ("mean_precision_without_background", "Precision"),
            ("mean_recall_without_background", "Recall"),
        ):
            epochs, values = series(metric, eval_metric=True)
            if values:
                axes[2].plot(epochs, values, marker="o", label=label)
        axes[2].set_title("Precision / Recall")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        for metric, label in (
            ("damage_false_positive_rate", "False positive rate"),
            ("damage_false_negative_rate", "False negative rate"),
            ("predicted_damage_pixel_ratio", "Predicted damage pixel ratio"),
        ):
            epochs, values = series(metric, eval_metric=True)
            if values:
                axes[3].plot(epochs, values, marker="o", label=label)
        axes[3].set_title("Damage Detection Behavior")
        axes[3].set_xlabel("epoch")
        axes[3].set_ylim(0, 1)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="best")

        epochs, values = series("learning_rate")
        if values:
            axes[4].plot(epochs, values, marker="o", label="learning rate")
        axes[4].set_title("Learning Rate")
        axes[4].set_xlabel("epoch")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(loc="best")

        class_labels = []
        class_ious = []
        class_dices = []
        for class_id in range(1, len(id2label)):
            metric_label = id2label[class_id].replace(" ", "_").replace("-", "_")
            iou_key = f"eval_iou_{metric_label}"
            dice_key = f"eval_dice_{metric_label}"
            if iou_key in best_eval:
                class_labels.append(id2label[class_id])
                class_ious.append(float(best_eval[iou_key]))
                class_dices.append(float(best_eval.get(dice_key, 0.0)))
        if class_labels:
            x_positions = np.arange(len(class_labels))
            width = 0.38
            axes[5].bar(x_positions - width / 2, class_ious, width, label="IoU")
            axes[5].bar(x_positions + width / 2, class_dices, width, label="Dice")
            axes[5].set_xticks(x_positions)
            axes[5].set_xticklabels(class_labels, rotation=30, ha="right")
        axes[5].set_title("Per-Class Metrics at Best mIoU Epoch")
        axes[5].set_ylim(0, 1)
        axes[5].grid(True, axis="y", alpha=0.3)
        axes[5].legend(loc="best")

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    resolved_run_name = run_name or _run_name_from_model(model_name)
    output_dir = f"/runs/{resolved_run_name}"
    shutil.rmtree(output_dir, ignore_errors=True)
    print(
        f"Training SegFormer model '{model_name}' on CarDD as run '{resolved_run_name}' "
        f"with {len(train_dataset)} train and {len(eval_dataset)} validation samples."
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="best",
        logging_strategy="epoch",
        logging_first_step=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou_without_background",
        greater_is_better=True,
        fp16=False,
        bf16=True,
        dataloader_num_workers=8,
        disable_tqdm=True,
        remove_unused_columns=False,
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    log_history_path = f"{output_dir}/trainer_log_history.json"
    with open(log_history_path, "w") as log_file:
        json.dump(trainer.state.log_history, log_file, indent=2)
    plot_path = f"{output_dir}/training_summary.png"
    save_training_plots(trainer.state.log_history, plot_path)
    print(f"Saved training curves and validation metric plots to {plot_path}")

    final_model_dir = f"{output_dir}/model"
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    with open(f"{final_model_dir}/metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"Training complete. Saved Hugging Face SegFormer model to {final_model_dir}")
    output_volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = "nvidia/mit-b2",
    epochs: int = 50,
    image_size: int = 512,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 6e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    run_name: str = "",
):
    resolved_run_name = run_name or _run_name_from_model(model_name)
    print(f"Dispatching SegFormer car-damage segmentation training job for {model_name}...")
    train.remote(
        model_name,
        epochs,
        image_size,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
        weight_decay,
        seed,
        run_name,
    )
    print("To retrieve the trained Hugging Face model directory after it finishes, run:")
    print(
        "modal volume get car-damage-segformer-output-vol "
        f"/{resolved_run_name}/model ./models/car_damage_segformer"
    )
