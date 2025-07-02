import os
import cv2
import torch
import numpy as np
from PIL import Image
from gtts import gTTS
from IPython.display import Audio, display
from ultralytics import YOLO

# Load YOLOv10s model (requires ultralytics >= 8.1.0)
model = YOLO('yolov10s.pt')  # Make sure yolov10s.pt is in your working directory

# === Text-to-Speech ===
def speak(text, filename='instruction.mp3'):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        display(Audio(filename, autoplay=True))
    except Exception as e:
        print(f"Error generating or playing audio: {e}")
        print(f"Spoken instruction: {text}")

# === Traffic Light Color Detection ===
def detect_light_color(crop):
    if crop is None or crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels and red_pixels > 100:
        return "Red"
    elif green_pixels > red_pixels and green_pixels > 100:
        return "Green"
    else:
        return "Unknown"

# === Main Logic ===
def provide_instruction(image_path):
    if not os.path.exists(image_path):
        print("Image not found.")
        speak("Image file not found.")
        return

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    results = model(image_path, conf=0.25)[0]

    traffic_lights = []
    vehicles_detected = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == 'traffic light':
            crop = img[y1:y2, x1:x2]
            color = detect_light_color(crop)
            center_x = (x1 + x2) // 2
            position = "center"
            if center_x < width // 3:
                position = "left"
            elif center_x > 2 * width // 3:
                position = "right"

            traffic_lights.append((color, position, (x1, y1, x2, y2)))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"TL: {color}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif label in ['car', 'bus', 'truck', 'motorcycle']:
            vehicles_detected += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Instruction logic
    message = ""
    if not traffic_lights:
        message = "No traffic light detected. Please be cautious."
    else:
        messages = []
        for color, position, _ in traffic_lights:
            if color == "Red":
                messages.append(f"A red traffic light is on your {position}. Please stop.")
            elif color == "Green":
                if vehicles_detected:
                    messages.append(f"A green traffic light is on your {position}, but vehicles are ahead. Please wait.")
                else:
                    messages.append(f"A green traffic light is on your {position}. You can go.")
            else:
                messages.append(f"Traffic light color is unclear on your {position}. Please wait or ask for help.")

        message = " ".join(messages)

    if vehicles_detected:
        message += f" {vehicles_detected} vehicle(s) detected ahead."

    print("🧠 Instruction:", message)
    speak(message)

    cv2.imwrite("annotated_output_YOLOv10.jpg", img)
    print("🖼️ Annotated image saved to annotated_output_YOLOv10.jpg")

# === Run ===
if __name__ == "__main__":
    image_path = input("Enter path to image: ").strip()
    provide_instruction(image_path)