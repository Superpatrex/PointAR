import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

qr_detector = cv2.QRCodeDetector()

def get_center(points):
    return np.mean(points.reshape(-1, 2), axis=0)

def is_pointing_at(qr_center, finger_base, finger_tip,
                   max_perp_dist=30, max_forward_dist=400):
    """
    qr_center, finger_base, finger_tip: np.array([x, y])
    max_perp_dist: how close to the ray (pixels)
    max_forward_dist: how far along the ray we care (pixels)
    """
    # Direction of finger
    d = finger_tip - finger_base
    norm = np.linalg.norm(d)
    if norm < 1e-5:
        return False
    d = d / norm  # normalize

    v = qr_center - finger_base

    # Projection of v onto d (scalar)
    t = np.dot(v, d)

    # If t < 0, QR is "behind" the finger
    if t < 0 or t > max_forward_dist:
        return False

    # Perpendicular distance from point to ray
    closest_point = finger_base + t * d
    perp_dist = np.linalg.norm(qr_center - closest_point)

    return perp_dist < max_perp_dist

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect QR code
    data, points, _ = qr_detector.detectAndDecode(frame)
    qr_center = None

    if points is not None:
        pts = points[0]
        qr_center = get_center(pts)
        pts = pts.astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.circle(frame, tuple(qr_center.astype(int)), 5, (0, 255, 0), -1)

    # 2. Detect hand + index finger
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_base = None
    finger_tip = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Base of index (landmark 5)
            lm_base = hand_landmarks.landmark[5]
            finger_base = np.array([lm_base.x * w, lm_base.y * h])

            cv2.circle(frame, tuple(finger_base.astype(int)), 6, (255, 0, 0), -1)

            # Draw the ray line for visualization
            direction = finger_base
            if np.linalg.norm(direction) > 1e-5:
                direction = direction / np.linalg.norm(direction)
                end_point = finger_base + direction * 500  # long ray
                cv2.line(frame,
                         tuple(finger_base.astype(int)),
                         tuple(end_point.astype(int)),
                         (255, 255, 0), 2)

    # 3. Check if the finger ray is pointing at the QR code
    if qr_center is not None and finger_base is not None and finger_tip is not None:
        if is_pointing_at(qr_center, finger_base, finger_tip):
            print("Ray hits QR!")
        else:
            print("Not pointing")

    cv2.imshow("Ray from Index Finger to QR", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
