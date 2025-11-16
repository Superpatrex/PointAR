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
# This should be the BASE URL, e.g., "http://192.168.1.100:5000"
# RASPBERRY_PI_URL = os.getenv("RASPBERRY_PI_URL")
# if not RASPBERRY_PI_URL:
#     print("[ERROR] RASPBERRY_PI_URL not set in .env file. Exiting.")
#     exit()

# --- NEW: RASPBERRY PI CONFIG for multiple IPs ---
LAMP_URLS = {
    "lamp-1": os.getenv("RASPBERRY_PI_URL_LAMP_1"),
    "lamp-2": os.getenv("RASPBERRY_PI_URL_LAMP_2")
}

# Check if URLs are loaded
if not LAMP_URLS["lamp-1"] or not LAMP_URLS["lamp-2"]:
    print("[ERROR] RASPBERRY_PI_URL_LAMP_1 or RASPBERRY_PI_URL_LAMP_2 not set in .env file. Exiting.")
    exit()

# List of valid QR codes that map to endpoints
VALID_QR_TARGETS = ["lamp-1", "lamp-2"] # This list is now implicitly handled by the dict keys
COOLDOWN_SECONDS = 2        # Prevent spamming requests

# ----------------- LOAD MODELS -----------------

# YOLOv8 custom-trained on "lamp"
model = YOLO(MODEL_PATH)

# NEW: QR Code Detector
# qr_detector = cv2.QRCodeDetector() # <-- REMOVED: We are using pyzbar now

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
hover_id = None # <-- MODIFIED: Tracks specific ID ("lamp-1") instead of "lamp"
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

# --- Helper functions for Ray-Tracing ---

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

    # ----- 1. Run YOLO object detection AND QR Code Reading -----
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
            # --- NEW: If it's a lamp, try to read QR code INSIDE the box ---
            if label == "lamp":
                # Add padding to ROI to help QR detector
                pad = 20
                lamp_roi = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                
                if lamp_roi.size > 0:
                    try:
                        # 1. Convert to grayscale
                        gray_roi = cv2.cvtColor(lamp_roi, cv2.COLOR_BGR2GRAY)
                        
                        # --- NEW: Advanced Preprocessing for Glare/Overexposure ---
                        
                        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                        # This enhances local contrast and is excellent for uneven lighting/glare.
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        clahe_roi = clahe.apply(gray_roi)

                        # 3. Apply a gentle blur to reduce noise from CLAHE
                        blur_roi = cv2.GaussianBlur(clahe_roi, (5, 5), 0)

                        # 4. Apply Adaptive Thresholding
                        # This is the main attempt to get a clean black-and-white image
                        thresh_roi = cv2.adaptiveThreshold(
                            blur_roi, 255, 
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 
                            21, # Block size (must be odd)
                            3   # C value (subtracted from the mean)
                        )
                        # --- END NEW LOGIC ---
                        
                        # 5. Attempt decoding in order of likelihood
                        
                        # Attempt 1: Decode the clean, thresholded image
                        barcodes = pyzbar_decode(thresh_roi)
                        
                        if not barcodes:
                            # Fallback 1: Try decoding the inverted threshold image
                            # (Sometimes glare inverts the image, making lines lighter)
                            thresh_roi_inv = cv2.adaptiveThreshold(
                                blur_roi, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, # <-- Inverted
                                21, 3
                            )
                            barcodes = pyzbar_decode(thresh_roi_inv)
                        
                        if not barcodes:
                            # Fallback 2: Try decoding the CLAHE-enhanced image directly
                            # (Maybe thresholding removed too much info)
                            barcodes = pyzbar_decode(clahe_roi)

                        if not barcodes:
                            # Fallback 3: Try decoding the original grayscale ROI
                            # (The simplest case)
                            barcodes = pyzbar_decode(gray_roi)
                        
                        if barcodes:
                            data = barcodes[0].data.decode('utf-8')
                            if data:
                                qr_data = data
                        # data, _, _ = qr_detector.detectAndDecode(lamp_roi) # <-- REMOVED
                        # if data:
                        #     qr_data = data
                    except Exception as e:
                        print(f"QR Error: {e}") # Print the error
                        pass # QR detection can fail, just ignore it
            
            detections.append((x1, y1, x2, y2, label, conf, qr_data))

            # --- MODIFIED: Draw detections with QR data if available ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if qr_data:
                text = f"{qr_data} ({conf:.2f})"
            else:
                text = f"{label} ({conf:.2f})"
                
            cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

    # ----- 2. Run MediaPipe Hands to get pointing ray (Unchanged) -----
    ray_base_xy = None
    ray_tip_xy = None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False # Optimize
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        ray_base_xy = np.array([lm_base.x * w, lm_base.y * h])
        lm_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ray_tip_xy = np.array([lm_tip.x * w, lm_tip.y * h])

        ray_dir = ray_tip_xy - ray_base_xy
        if np.linalg.norm(ray_dir) > 1e-5:
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            end_point = ray_base_xy + ray_dir * 1500
            cv2.line(frame, tuple(ray_base_xy.astype(int)), tuple(end_point.astype(int)), (255, 255, 0), 2)

    # ----- 3. Finger-as-cursor selection logic (MODIFIED for QR ID) -----
    status_text = "No hand or no objects"
    status_color = (0, 255, 255)

    if ray_base_xy is not None and ray_tip_xy is not None and detections:
        hovered = None
        best_conf = 0.0

        for (x1, y1, x2, y2, label, conf, qr_data) in detections:
            if label == "lamp": # Only interact with lamps
                target_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                target_diameter = ((x2 - x1) + (y2 - y1)) / 2
                
                if is_pointing_at(target_center, ray_base_xy, ray_tip_xy,
                                    target_diameter=target_diameter):
                    if conf > best_conf:
                        best_conf = conf
                        hovered = (x1, y1, x2, y2, label, conf, qr_data)

        if hovered is not None:
            x1, y1, x2, y2, label, conf, qr_data = hovered
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Highlight

            # --- MODIFIED: Track by specific QR ID ---
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
                    
                    # --- MODIFIED: Send request to specific endpoint ---
                    # Check if the detected QR ID has a URL mapped to it
                    url_to_send = LAMP_URLS.get(hover_id)
                    
                    if url_to_send:
                        current_time = time.time()
                        if (current_time - last_request_time) > COOLDOWN_SECONDS:
                            
                            # Construct the final URL, e.g. "http://base_url/lamp-1"
                            # final_url = f"{RASPBERRY_PI_URL.rstrip('/')}/{hover_id}"
                            
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