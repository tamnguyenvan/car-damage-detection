import os
import sys
import cv2
import requests
import argparse

# Cấu hình API Endpoint (đảm bảo Docker container đang chạy ở port 8000)
API_URL = "http://localhost:8000/predict"

def process_and_visualize(image_path: str, save_result: bool = False):
    """
    Sends an image to the YOLOv8 API, parses the response, 
    and uses OpenCV to visualize the bounding boxes.
    """
    # 1. Check if file exists
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    print(f"[INFO] Sending '{image_path}' to {API_URL}...")

    # 2. Prepare and send the POST request
    with open(image_path, "rb") as image_file:
        files = {
            "file": (os.path.basename(image_path), image_file, "image/jpeg")
        }
        try:
            response = requests.post(API_URL, files=files)
        except requests.exceptions.ConnectionError:
            print("[ERROR] Could not connect to API. Is the Docker container running?")
            sys.exit(1)

    # 3. Handle API response
    if response.status_code != 200:
        print(f"[ERROR] API returned status code {response.status_code}")
        print(f"Details: {response.text}")
        sys.exit(1)

    data = response.json()
    if not data.get("success"):
        print(f"[ERROR] API processed the request but returned a failure: {data.get('error')}")
        sys.exit(1)

    detections = data.get("detections", [])
    print(f"[INFO] API Success! Detected {len(detections)} damage instance(s).")

    # 4. Load the original image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] OpenCV could not read the image file.")
        sys.exit(1)

    # 5. Draw bounding boxes and labels
    for det in detections:
        box = det["box"]           # [x1, y1, x2, y2]
        conf = det["confidence"]   # float
        cls_name = det["class_name"] # str

        # Convert float coordinates to integers for cv2
        x1, y1, x2, y2 = map(int, box)
        
        # Define colors and labels
        color = (0, 0, 255) # Red color in BGR
        label = f"{cls_name} ({conf:.2f})"
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw label background for better text visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw filled rectangle for text background
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, thickness=-1)
        
        # Put text
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    # 6. Save the output image
    if save_result:
        output_filename = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, img)
        print(f"[INFO] Saved visualization to: {output_filename}")

    # 7. Display the image in a window (Might fail on headless servers)
    try:
        cv2.imshow("Car Damage Detection", img)
        print("[INFO] Press any key on the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("[WARNING] Could not open GUI window (normal for server/WSL environments).")
        if save_result:
            print(f"Please check the saved file '{output_filename}' instead.")
        else:
            print("You can run with --save to save the output image instead.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Car Damage Detection API")
    parser.add_argument("image_path", type=str, help="Path to the test image")
    parser.add_argument("--save", action="store_true", help="Save the visualization result to a file")
    
    args = parser.parse_args()
    
    process_and_visualize(args.image_path, save_result=args.save)