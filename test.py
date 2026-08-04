import argparse
import colorsys
import hashlib
import mimetypes
import os
import sys

import cv2
import numpy as np
import requests

API_URL = "http://localhost:8000/predict"
WINDOW_NAME = "Car Damage Assessment"
LEGEND_BG_COLOR = (0, 0, 0)

# Pre-defined palette — must match DAMAGE_CLASSES from the training script.
# Colors chosen to stand out against typical vehicle colors
# (white, black, silver, gray, red, blue, etc.).
_PREDEFINED_COLORS: dict[str, tuple[int, int, int]] = {
    "crack":         (255, 0, 255),     # Magenta / Fuchsia
    "dent":          (255, 255, 0),     # Cyan
    "scratch":       (0, 255, 50),      # Lime / Neon Green
    "glass shatter": (220, 0, 220),     # Bright Purple
    "lamp broken":   (100, 140, 255),   # Coral / Salmon
    "tire flat":     (200, 255, 0),     # Turquoise
}


def _get_damage_color(class_name: str) -> tuple[int, int, int]:
    """Return a consistent, distinct color for any damage class name via hashing."""
    key = class_name.lower().strip()
    if key in _PREDEFINED_COLORS:
        return _PREDEFINED_COLORS[key]

    # Hash the class name to generate stable, well-separated RGB values
    digest = hashlib.md5(key.encode()).digest()
    # Use 3 bytes for hue-ish spacing, ensure vivid colors (avoid grey/muddy)
    h = digest[0] / 255.0                     # hue-like 0-1
    s = 0.55 + (digest[1] / 255.0) * 0.45    # saturation 0.55-1.0 (vivid)
    v = 0.60 + (digest[2] / 255.0) * 0.40    # value 0.60-1.0 (not too dark)

    # HSV → RGB (BGR for OpenCV)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def _format_class_name(name: str) -> str:
    """Convert snake_case or lowercase to Title Case for display."""
    return name.replace("_", " ").title()


def _draw_damage_mask(image, polygon, color: tuple[int, int, int]) -> None:
    """Draw a prominent damage mask overlay."""
    if not polygon:
        return

    points = np.asarray(polygon, dtype=np.int32)
    if points.ndim != 2 or len(points) < 3 or points.shape[1] != 2:
        return

    # Semi-transparent fill
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.42, image, 0.58, 0, image)


def _draw_legend(image, detections: list[dict], position: str = "top-right") -> None:
    """Draw a legend panel at the chosen corner: top-left, top-right, bottom-left, bottom-right."""
    image_height, image_width = image.shape[:2]

    # Collect unique (damage, part) pairs with their colors
    entries: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[tuple[str, str]] = set()
    for det in detections:
        dn = det.get("damage_label") or det["class_name"]
        damage_name = _format_class_name(dn)
        part_name = _format_class_name(det.get("car_part") or "unknown part")
        key = (damage_name, part_name)
        if key not in seen:
            seen.add(key)
            label = f"{damage_name}  -  {part_name}"
            entries.append((label, _get_damage_color(dn)))

    if not entries:
        return

    entries.sort(key=lambda x: x[0])

    # ------ Panel layout: right side, sized to content ------
    font = cv2.FONT_HERSHEY_DUPLEX
    font_thickness = 2
    pad = max(8, image_width // 100)
    swatch_sz = max(18, image_height // 28)
    gap = max(6, image_width // 160)

    # Determine font scale by fitting the longest label within a max text width budget.
    # The panel itself can be at most ~1/3 of image width; the text portion is
    # panel_w - (pad + swatch_sz + gap + pad).  We target ~28% of image width for
    # the text to leave headroom for the swatch and padding.
    max_text_width = int(image_width * 0.25)

    item_scale = 1.0
    max_label_width = 0
    max_label_height = 0
    for label, _ in entries:
        for scale in (0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35):
            (tw, th_item), baseline = cv2.getTextSize(label, font, scale, font_thickness)
            if tw <= max_text_width:
                max_label_width = max(max_label_width, tw)
                max_label_height = max(max_label_height, th_item + baseline)
                item_scale = min(item_scale, scale)
                break
        else:
            # Fallback – use the smallest scale we're willing to render
            (tw, th_item), baseline = cv2.getTextSize(label, font, 0.35, font_thickness)
            max_label_width = max(max_label_width, tw)
            max_label_height = max(max_label_height, th_item + baseline)
            item_scale = min(item_scale, 0.35)

    # Panel width = left-pad + swatch + gap + max-text-width + right-pad
    panel_w = pad + swatch_sz + gap + max_label_width + pad
    # Clamp: at least 1/6 of image width, at most 1/3
    panel_w = max(image_width // 6, min(panel_w, image_width // 3))

    # Title
    title = "LEGEND"
    title_scale = min(item_scale * 1.2, panel_w / 280.0, 0.75)
    (tw_title, th_title), _ = cv2.getTextSize(title, font, title_scale, font_thickness)
    title_y = pad + th_title
    divider_y = title_y + max(6, image_height // 70)

    # Row height: the larger of the swatch or the text block (text height + a
    # small gap so rows never touch). This guarantees text is never clipped.
    row_text_height = max_label_height + max(3, image_height // 120)
    row_h = max(swatch_sz, row_text_height)
    # Extra vertical gap between rows for readability
    row_spacing = row_h + max(2, image_height // 160)

    # Total panel height
    panel_h = divider_y + pad + len(entries) * row_spacing + pad

    # Position the panel in the chosen corner
    position = position.lower().replace("_", "-")
    if position in ("top-right", "right-top"):
        px = image_width - panel_w - pad
        py = pad
    elif position in ("top-left", "left-top"):
        px = pad
        py = pad
    elif position in ("bottom-right", "right-bottom"):
        px = image_width - panel_w - pad
        py = image_height - panel_h - pad
    elif position in ("bottom-left", "left-bottom"):
        px = pad
        py = image_height - panel_h - pad
    else:
        # Fallback to top-right
        px = image_width - panel_w - pad
        py = pad

    # Dark background
    overlay = image.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                  LEGEND_BG_COLOR, thickness=-1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    # Subtle border
    cv2.rectangle(image, (px, py), (px + panel_w, py + panel_h), (80, 80, 80), 1)

    # Title – centered horizontally
    title_x = px + (panel_w - tw_title) // 2
    cv2.putText(image, title, (title_x, title_y), font, title_scale,
                (220, 220, 220), font_thickness, lineType=cv2.LINE_AA)

    # Divider line under title
    cv2.line(image, (px + pad, divider_y), (px + panel_w - pad, divider_y),
             (100, 100, 100), 1)

    # Damage entries
    cur_y = divider_y + pad
    for name, color in entries:
        # Color swatch – vertically centred within row_h
        swatch_y = cur_y + (row_h - swatch_sz) // 2
        cv2.rectangle(image, (px + pad, swatch_y),
                      (px + pad + swatch_sz, swatch_y + swatch_sz), color,
                      thickness=-1)
        cv2.rectangle(image, (px + pad, swatch_y),
                      (px + pad + swatch_sz, swatch_y + swatch_sz),
                      (230, 230, 230), 1)

        # Text – baseline aligned to swatch centre for a natural look
        (tw_item, th_item), baseline = cv2.getTextSize(name, font, item_scale,
                                                       font_thickness)
        text_x = px + pad + swatch_sz + gap
        # Place text so its vertical centre aligns with the swatch centre.
        # cv2.putText positions by the text *baseline* (bottom-left corner).
        # We want:    baseline = swatch_mid  +  (th_item / 2)
        text_y = cur_y + row_h // 2 + th_item // 2
        cv2.putText(image, name, (text_x, text_y), font, item_scale,
                    (245, 245, 245), font_thickness, lineType=cv2.LINE_AA)

        cur_y += row_spacing


def process_and_visualize(
    image_path: str,
    save_result: bool = False,
    window_width: int = 1280,
    window_height: int = 800,
    show_window: bool = True,
    legend_position: str = "top-right",
) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"[INFO] Sending '{image_path}' to {API_URL}...")
    content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as image_file:
        response = requests.post(
            API_URL,
            params={"response_mode": "full"},
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
        damage_name = detection.get("damage_label") or detection["class_name"]
        damage_count = int(detection.get("damage_count") or 1)
        count_details = f", merged={damage_count}" if damage_count > 1 else ""
        print(
            f"  {index}. damage={damage_name} ({detection['confidence']:.2f}{count_details}), "
            f"car part={car_part}{match_details}"
        )
        # Draw damage area with prominent highlighting
        _draw_damage_mask(image, detection.get("damage_polygon"),
                          _get_damage_color(damage_name))

    # Draw legend
    _draw_legend(image, detections, position=legend_position)

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
    parser.add_argument("--legend-position", default="top-right",
                        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
                        help="Corner to place the legend panel")
    args = parser.parse_args()

    try:
        process_and_visualize(
            args.image_path,
            save_result=args.save,
            window_width=args.window_width,
            window_height=args.window_height,
            show_window=not args.no_show,
            legend_position=args.legend_position,
        )
    except (FileNotFoundError, requests.RequestException, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
