import cv2
import os

# --- Config ---
OUTPUT_DIR = "dataset_with_qr_codes_2"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL = "lamp"  # The object you are capturing
# --------------

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Find the next available image number
existing_files = [f for f in os.listdir(IMAGE_DIR) if f.startswith(LABEL)]
count = 0
if existing_files:
    numbers = [int(f.replace(f"{LABEL}_", "").replace(".jpg", "")) for f in existing_files]
    if numbers:
        count = max(numbers) + 1

cap = cv2.VideoCapture(0)
print("Starting camera feed...")
print("Press 's' to save an image.")
print("Press 'q' or 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow("Data Capture - Press 's' to save, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    # Save on 's'
    if key == ord('s'):
        filename = f"{LABEL}_{count:03d}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved: {filepath}")
        count += 1
    
    # Quit on 'q' or ESC
    elif key == ord('q') or key == 27:
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images in {IMAGE_DIR}")