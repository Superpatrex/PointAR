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
from pyzbar.pyzbar import decode as pyzbar_decode # <-- NEW: Import pyzbar

# ----------------- CONFIG -----------------

load_dotenv()

# Use the 'run13' model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(
    BASE_DIR / "lamp_detection" / "lamp_detector" / "run14" / "weights" / "best.pt"
)
CONF_THRESHOLD = 0.5        # YOLO confidence threshold
HIT_FRAMES = 4              # dwell frames needed to "select" an object

# --- RASPBERRY PI CONFIG ---
# --- NEW: RASPBERRY PI CONFIG for multiple IPs ---
LAMP_URLS = {
    "lamp-1": os.getenv("RASPBERRY_PI_URL_LAMP_1"),
    "lamp-2": os.getenv("RASPBERRY_PI_URL_LAMP_2")
}

# Check if URLs are loaded
if not LAMP_URLS["lamp-1"] or not LAMP_URLS["lamp-2"]:
    print("[ERROR] RASPBERRY_PI_URL_LAMP_1 or RASPBERRY_PI_URL_LAMP_2 not set in .env file. Exiting.")
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
    min_detection_confidence=0.3, # Lowered for responsiveness
    min_tracking_confidence=0.3  # Lowered for responsiveness
)
mp_draw = mp.solutions.drawing_utils # For drawing hand skeleton

# ----------------- STATE -----------------

cap = cv2.VideoCapture(0)
hover_id = None # Tracks specific ID ("lamp-1")
hover_frames = 0
selected_id = None # for debug display
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
        print(f"[ERROR] Could not connect to Raspberry Pi at {url}")
    except requests.Timeout:
        print(f"[ERROR] Request to Pi timed out.")
    except Exception as e:
        print(f"[ERROR] An unknown error occurred: {e}")

# --- REVERTED: Back to simple inside_box helper ---
def inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# ----------------- MAIN LOOP -----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ----- 1. Run YOLO object detection AND QR Code Reading (Unchanged) -----
    results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = [] # Will store (x1, y1, x2, y2, label, conf, qr_data)
    
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id]
            
            qr_data = None
            if label == "lamp":
                pad = 20
                lamp_roi = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                
                if lamp_roi.size > 0:
                    try:
                        # 1. Convert to grayscale
                        gray_roi = cv2.cvtColor(lamp_roi, cv2.COLOR_BGR2GRAY)
                        
                        # 2. Apply CLAHE
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        clahe_roi = clahe.apply(gray_roi)

                        # 3. Apply a gentle blur
                        blur_roi = cv2.GaussianBlur(clahe_roi, (5, 5), 0)

                        # 4. Apply Adaptive Thresholding
                        thresh_roi = cv2.adaptiveThreshold(
                            blur_roi, 255, 
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 
                            21, 3
                        )
                        
                        # 5. Attempt decoding in order
                        barcodes = pyzbar_decode(thresh_roi)
                        
                        if not barcodes:
                            thresh_roi_inv = cv2.adaptiveThreshold(
                                blur_roi, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV,
                                21, 3
                            )
                            barcodes = pyzbar_decode(thresh_roi_inv)
                        
                        if not barcodes:
                            barcodes = pyzbar_decode(clahe_roi)

                        if not barcodes:
                            barcodes = pyzbar_decode(gray_roi)
                        
                        if barcodes:
                            data = barcodes[0].data.decode('utf-8')
                            if data:
                                qr_data = data
                    except Exception as e:
                        print(f"QR Error: {e}")
                        pass
            
            detections.append((x1, y1, x2, y2, label, conf, qr_data))

            # Draw detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if qr_data:
                text = f"{qr_data} ({conf:.2f})"
            else:
                text = f"{label} ({conf:.2f})"
                
            cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

    # ----- 2. Run MediaPipe Hands to get fingertip (REVERTED) -----
    fingertip_xy = None # <-- REVERTED
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- REVERTED: Get only the index tip as a cursor ---
        INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
        lm_tip = hand_landmarks.landmark[INDEX_TIP]
        fx = int(lm_tip.x * w)
        fy = int(lm_tip.y * h)
        fingertip_xy = (fx, fy)
        cv2.circle(frame, fingertip_xy, 8, (0, 0, 255), -1) # Draw cursor
        # --- End Revert ---

    # ----- 3. Finger-as-cursor selection logic (MODIFIED for inside_box) -----
    status_text = "No hand or no objects"
    status_color = (0, 255, 255)

    if fingertip_xy is not None and detections: # <-- REVERTED
        fx, fy = fingertip_xy # <-- REVERTED
        hovered = None
        best_conf = 0.0

        for (x1, y1, x2, y2, label, conf, qr_data) in detections:
            if label == "lamp": # Only interact with lamps
                
                # --- REVERTED: Use inside_box check ---
                if inside_box(fx, fy, (x1, y1, x2, y2)):
                # --- End Revert ---
                    if conf > best_conf:
                        best_conf = conf
                        hovered = (x1, y1, x2, y2, label, conf, qr_data)

        if hovered is not None:
            x1, y1, x2, y2, label, conf, qr_data = hovered
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Highlight

            # --- MODIFIED: Track by specific QR ID (Unchanged) ---
            current_id = qr_data if qr_data else "unknown_lamp"

            if hover_id == current_id:
                hover_frames += 1
            else:
                hover_id = current_id
                hover_frames = 1

            if hover_frames >= HIT_FRAMES:
                status_text = f"SELECTED: {hover_id}"
                status_color = (0, 0, 255)
                selected_id = hover_id

                if hover_frames == HIT_FRAMES:
                    print(f"[ACTION] Selected object: {hover_id}")
                    
                    # --- Send request logic (Unchanged) ---
                    url_to_send = LAMP_URLS.get(hover_id)
                    
                    if url_to_send:
                        current_time = time.time()
                        if (current_time - last_request_time) > COOLDOWN_SECONDS:
                            print(f"[ACTION] Sending request to Pi: {url_to_send}")
                            threading.Thread(target=send_pi_request, args=(url_to_send,)).start()
                            last_request_time = current_time
                        else:
                            print("[ACTION] Cooldown active, not sending request.")
                    else:
                        print(f"[INFO] Selected lamp, but QR ID '{hover_id}' has no URL configured.")
            else:
                status_text = f"Hovering over: {hover_id}"
                status_color = (0, 255, 255)
        else:
            hover_id = None
            hover_frames = 0
    else:
        hover_id = None
        hover_frames = 0

    # ----- 4. Overlay status (Unchanged) -----
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

    if selected_id is not None:
        cv2.putText(
            frame,
            f"Last selected: {selected_id}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Finger Cursor + YOLO + QR Selection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()