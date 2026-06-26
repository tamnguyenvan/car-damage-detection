import argparse
import mimetypes
import os
import sys

import cv2
import numpy as np
import requests

API_URL = "http://localhost:8000/predict"
WINDOW_NAME = "Car Damage Assessment"
DAMAGE_COLOR = (36, 39, 235)
DAMAGE_MASK_COLOR = (0, 140, 255)
PART_MASK_COLOR = (42, 180, 42)
TEXT_COLOR = (255, 255, 255)


def _draw_label(image, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 5
    line_height = 19
    image_height, image_width = image.shape[:2]
    text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    text_height = line_height * len(lines) + padding * 2

    label_x = max(0, min(x, image_width - text_width - padding * 2))
    label_y = y - text_height - 3
    if label_y < 0:
        label_y = min(image_height - text_height, y + 3)

    cv2.rectangle(
        image,
        (label_x, label_y),
        (label_x + text_width + padding * 2, label_y + text_height),
        color,
        thickness=-1,
    )
    for index, line in enumerate(lines):
        text_y = label_y + padding + (index + 1) * line_height - 4
        cv2.putText(image, line, (label_x + padding, text_y), font, font_scale, TEXT_COLOR, thickness)


def _draw_detection(image, detection: dict) -> None:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    damage_label = f"Damage: {detection['class_name']} {detection['confidence']:.2f}"
    car_part = detection.get("car_part")
    if car_part:
        part_confidence = detection.get("part_confidence")
        coverage = detection.get("part_coverage")
        iou = detection.get("part_iou")
        part_label = f"Part: {car_part}"
        if part_confidence is not None:
            part_label += f" {part_confidence:.2f}"
        if coverage is not None:
            part_label += f" | coverage {coverage:.2f}"
        if iou is not None:
            part_label += f" | IoU {iou:.2f}"
    else:
        part_label = "Part: no match"

    cv2.rectangle(image, (x1, y1), (x2, y2), DAMAGE_COLOR, thickness=2)
    _draw_label(image, [damage_label, part_label], x1, y1, DAMAGE_COLOR)


def _draw_segmentation(image, polygon, color: tuple[int, int, int]) -> None:
    if not polygon:
        return

    points = np.asarray(polygon, dtype=np.int32)
    if points.ndim != 2 or len(points) < 3 or points.shape[1] != 2:
        return

    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0, image)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)


def process_and_visualize(
    image_path: str,
    save_result: bool = False,
    window_width: int = 1280,
    window_height: int = 800,
    show_window: bool = True,
) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"[INFO] Sending '{image_path}' to {API_URL}...")
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as image_file:
        response = requests.post(
            API_URL,
            files={"file": (os.path.basename(image_path), image_file, content_type)},
            timeout=60,
        )

    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error") or "API returned an unsuccessful response.")

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image file: {image_path}")

    detections = data.get("detections", [])
    print(f"[INFO] Detected {len(detections)} damage instance(s).")
    for index, detection in enumerate(detections, start=1):
        car_part = detection.get("car_part") or "no matched part"
        part_confidence = detection.get("part_confidence")
        coverage = detection.get("part_coverage")
        iou = detection.get("part_iou")
        match_details = ""
        if part_confidence is not None:
            match_details += f", part confidence={part_confidence:.2f}"
        if coverage is not None:
            match_details += f", coverage={coverage:.2f}"
        if iou is not None:
            match_details += f", IoU={iou:.2f}"
        print(
            f"  {index}. damage={detection['class_name']} ({detection['confidence']:.2f}), "
            f"car part={car_part}{match_details}"
        )
        _draw_segmentation(image, detection.get("damage_polygon"), DAMAGE_MASK_COLOR)
        _draw_segmentation(image, detection.get("car_part_polygon"), PART_MASK_COLOR)
        _draw_detection(image, detection)

    output_filename = ""
    if save_result:
        output_filename = f"result_{os.path.basename(image_path)}"
        if not cv2.imwrite(output_filename, image):
            raise RuntimeError(f"Failed to save visualization to: {output_filename}")
        print(f"[INFO] Saved visualization to: {output_filename}")

    if not show_window:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, window_width, window_height)
    cv2.imshow(WINDOW_NAME, image)
    print("[INFO] Press any key in the visualization window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize car damage and matched car parts from the API")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--save", action="store_true", help="Save the annotated image as result_<filename>")
    parser.add_argument("--no-show", action="store_true", help="Do not open the OpenCV visualization window")
    parser.add_argument("--window-width", type=int, default=1280, help="Initial OpenCV window width")
    parser.add_argument("--window-height", type=int, default=800, help="Initial OpenCV window height")
    args = parser.parse_args()

    try:
        process_and_visualize(
            args.image_path,
            save_result=args.save,
            window_width=args.window_width,
            window_height=args.window_height,
            show_window=not args.no_show,
        )
    except (FileNotFoundError, requests.RequestException, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
