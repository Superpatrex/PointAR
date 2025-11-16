import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
import requests
import threading
import time

# ----------------- CONFIG -----------------

load_dotenv()

# Use the 'run2' model you trained to have better accuracy
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(
    # NOTE: You are using 'run12', make sure this is correct!
    BASE_DIR / "lamp_detection" / "lamp_detector" / "run13" / "weights" / "best.pt"
)
CONF_THRESHOLD = 0.5        # We can use a lower threshold now
HIT_FRAMES = 8              # dwell frames needed to "select" an object

# --- RASPBERRY PI CONFIG ---
RASPBERRY_PI_URL = os.getenv("RASPBERRY_PI_URL")
if not RASPBERRY_PI_URL:
    print("[ERROR] RASPBERRY_PI_URL not set in .env file. Exiting.")
    exit()
COOLDOWN_SECONDS = 2        # Prevent spamming requests

# ----------------- LOAD MODELS -----------------

# YOLOv8 custom-trained on "lamp"
model = YOLO(MODEL_PATH)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3, # <-- UPDATED: Lowered for responsiveness
    min_tracking_confidence=0.3  # <-- UPDATED: Lowered for responsiveness
)
mp_draw = mp.solutions.drawing_utils # <-- NEW: For drawing hand skeleton

# ----------------- STATE -----------------

cap = cv2.VideoCapture(0)
hover_label = None
hover_frames = 0
selected_label = None
last_request_time = 0

# ----------------- HELPER FUNCTIONS -----------------

def send_pi_request(url):
    """
    Sends an HTTP GET request in a separate thread
    to avoid freezing the main CV loop.
    """
    try:
        response = requests.get(url, timeout=3)
        print(f"[ACTION] Pi responded with: {response.status_code}")
    except requests.ConnectionError:
        print("[ERROR] Could not connect to Raspberry Pi.")
    except requests.Timeout:
        print("[ERROR] Request to Pi timed out.")
    except Exception as e:
        print(f"[ERROR] An unknown error occurred: {e}")

# --- NEW: Helper functions for Ray-Tracing ---

def get_center(points):
    """Calculates the center of a set of 2D points."""
    return np.mean(points.reshape(-1, 2), axis=0)

def is_pointing_at(target_center, ray_base, ray_tip,
                       target_diameter=None,
                       default_perp_dist=50,
                       min_perp_dist=25,
                       max_forward_dist=1500):
    """
    Checks if a 2D ray (from ray_base towards ray_tip) "hits" a target.
    """
    ray_dir = ray_tip - ray_base
    norm = np.linalg.norm(ray_dir)
    if norm < 1e-5:
        return False
    ray_dir = ray_dir / norm

    v_to_target = target_center - ray_base
    t = np.dot(v_to_target, ray_dir)

    if t < 0 or t > max_forward_dist:
        return False

    if target_diameter is not None:
        max_perp_dist = target_diameter * 1.0
        max_perp_dist = max(max_perp_dist, min_perp_dist)
    else:
        max_perp_dist = default_perp_dist

    closest_point = ray_base + t * ray_dir
    perp_dist = np.linalg.norm(target_center - closest_point)

    return perp_dist < max_perp_dist

# ----------------- MAIN LOOP -----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----- 1. Run YOLO object detection (Same as your file) -----
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

    # ----- 2. Run MediaPipe Hands to get pointing ray -----
    # <-- MODIFIED: Now gets ray_base and ray_tip
    ray_base_xy = None
    ray_tip_xy = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False # Optimize
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        
        # Draw the hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Get WRIST (landmark 0) as the base of the ray
        lm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        ray_base_xy = np.array([lm_base.x * w, lm_base.y * h])

        # Get INDEX FINGER TIP (landmark 8) as the tip of the ray
        lm_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ray_tip_xy = np.array([lm_tip.x * w, lm_tip.y * h])

        # Draw the cyan ray line for visualization
        ray_dir = ray_tip_xy - ray_base_xy
        if np.linalg.norm(ray_dir) > 1e-5:
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            end_point = ray_base_xy + ray_dir * 1500 # 1500 pixels long
            cv2.line(frame,
                     tuple(ray_base_xy.astype(int)),
                     tuple(end_point.astype(int)),
                     (255, 255, 0), 2) # Cyan ray

    # ----- 3. Finger-as-cursor selection logic -----
    # <-- MODIFIED: Using is_pointing_at() instead of inside_box()
    status_text = "No hand or no objects"
    status_color = (0, 255, 255)

    if ray_base_xy is not None and ray_tip_xy is not None and detections:
        hovered = None
        best_conf = 0.0

        for (x1, y1, x2, y2, label, conf) in detections:
            # --- NEW RAY-TRACING LOGIC ---
            target_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            target_diameter = ((x2 - x1) + (y2 - y1)) / 2 # Avg width/height
            
            if is_pointing_at(target_center, ray_base_xy, ray_tip_xy,
                                target_diameter=target_diameter):
            # --- END NEW LOGIC ---
                if conf > best_conf:
                    best_conf = conf
                    hovered = (x1, y1, x2, y2, label, conf)

        # --- This Dwell-Time and HTTP logic is all the same as your file ---
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

                if hover_frames == HIT_FRAMES:
                    print(f"[ACTION] Selected object: {label}")
                    if label == "lamp":
                        current_time = time.time()
                        if (current_time - last_request_time) > COOLDOWN_SECONDS:
                            print(f"[ACTION] Sending request to Pi: {RASPBERRY_PI_URL}")
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

    # ----- 4. Overlay status (Same as your file) -----
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