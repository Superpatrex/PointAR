import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
import requests  # <-- NEW: For HTTP requests
import threading # <-- NEW: To prevent video from freezing
import time      # <-- NEW: For request cooldown

# ----------------- CONFIG -----------------

load_dotenv()

# Use the 'run2' model you trained to have better accuracy
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(
    BASE_DIR / "lamp_detection" / "lamp_detector" / "run12" / "weights" / "best.pt"
)
CONF_THRESHOLD = 0.5        # We can use a lower threshold now
HIT_FRAMES = 8              # dwell frames needed to "select" an object

# --- NEW: RASPBERRY PI CONFIG ---
# !!! REPLACE with your Raspberry Pi's actual IP address and port !!!
# (Assuming a default Flask server port of 5000)
RASPBERRY_PI_URL = os.getenv("RASPBERRY_PI_URL")
COOLDOWN_SECONDS = 2        # Prevent spamming requests

# ----------------- LOAD MODELS -----------------

# YOLOv8 custom-trained on "lamp"
model = YOLO(MODEL_PATH)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------- STATE -----------------

cap = cv2.VideoCapture(0)

hover_label = None
hover_frames = 0
selected_label = None  # just for debug display

# --- NEW: State for request cooldown ---
last_request_time = 0


# --- NEW: HTTP Request Function ---

def send_pi_request(url):
    """
    Sends an HTTP GET request in a separate thread
    to avoid freezing the main CV loop.
    Includes error handling.
    """
    try:
        # timeout=3 means it will give up after 3 seconds
        response = requests.get(url, timeout=3)
        print(f"[ACTION] Pi responded with: {response.status_code}")
    except requests.ConnectionError:
        print("[ERROR] Could not connect to Raspberry Pi.")
    except requests.Timeout:
        print("[ERROR] Request to Pi timed out.")
    except Exception as e:
        print(f"[ERROR] An unknown error occurred: {e}")


def inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


# ----------------- MAIN LOOP -----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1) # Uncomment if you want to mirror
    h, w, _ = frame.shape

    # ----- 1. Run YOLO object detection -----
    results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id]
            detections.append((x1, y1, x2, y2, label, conf))

    # Draw detections
    for (x1, y1, x2, y2, label, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # ----- 2. Run MediaPipe Hands to get fingertip -----
    fingertip_xy = None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
        lm_tip = hand_landmarks.landmark[INDEX_TIP]
        fx = int(lm_tip.x * w)
        fy = int(lm_tip.y * h)
        fingertip_xy = (fx, fy)
        cv2.circle(frame, fingertip_xy, 8, (0, 0, 255), -1)

    # ----- 3. Finger-as-cursor selection logic -----
    status_text = "No hand or no objects"
    status_color = (0, 255, 255)

    if fingertip_xy is not None and detections:
        fx, fy = fingertip_xy
        hovered = None
        best_conf = 0.0

        for (x1, y1, x2, y2, label, conf) in detections:
            if inside_box(fx, fy, (x1, y1, x2, y2)):
                if conf > best_conf:
                    best_conf = conf
                    hovered = (x1, y1, x2, y2, label, conf)

        if hovered is not None:
            x1, y1, x2, y2, label, conf = hovered
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Highlight

            if hover_label == label:
                hover_frames += 1
            else:
                hover_label = label
                hover_frames = 1

            if hover_frames >= HIT_FRAMES:
                status_text = f"SELECTED: {label}"
                status_color = (0, 0, 255)
                selected_label = label

                # Trigger once when first selected
                if hover_frames == HIT_FRAMES:
                    print(f"[ACTION] Selected object: {label}")

                    # --- NEW: Check if it's the lamp and send request ---
                    if label == "lamp":
                        current_time = time.time()
                        if (current_time - last_request_time) > COOLDOWN_SECONDS:
                            print(f"[ACTION] Sending request to Pi: {RASPBERRY_PI_URL}")
                            # Run the request in a new thread
                            threading.Thread(target=send_pi_request, args=(RASPBERRY_PI_URL,)).start()
                            last_request_time = current_time
                        else:
                            print("[ACTION] Cooldown active, not sending request.")
            else:
                status_text = f"Hovering over: {label}"
                status_color = (0, 255, 255)
        else:
            hover_label = None
            hover_frames = 0
    else:
        hover_label = None
        hover_frames = 0

    # ----- 4. Overlay status -----
    cv2.putText(
        frame,
        status_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2,
        cv2.LINE_AA,
    )

    if selected_label is not None:
        cv2.putText(
            frame,
            f"Last selected: {selected_label}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Finger Cursor + YOLO Object Selection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
