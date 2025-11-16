from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# --- Your Specific Config ---
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is not set. Please add it to your .env file or shell environment."
    )
WORKSPACE_ID = "emoryhacks"
PROJECT_ID = "lamp-detector-lxnem"
VERSION_NUMBER = 3
# -----------------------------

# 1. Download Dataset from Roboflow
print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
dataset = project.version(VERSION_NUMBER).download("yolov8")

# The download gives us the path to the data.yaml file
data_yaml_path = os.path.join(dataset.location, "data.yaml")
print(f"Dataset downloaded to: {dataset.location}")
print(f"Using data.yaml from: {data_yaml_path}")

# 2. Load a pre-trained model to fine-tune
# We use 'yolov8n.pt' because it's fast, perfect for your project
model = YOLO('yolov8n.pt')

# 3. Train the model
print("Starting model training...")
results = model.train(
    data=data_yaml_path,  # Path to our dataset config
    epochs=50,            # 50 epochs is a good start
    imgsz=640,            # Match your inference size
    project="lamp_detector",
    name="run1"
)

print("Training complete!")
print("---" * 10)
print(f"Success! Your new model is saved in: {results.save_dir}/weights/best.pt")
print("---" * 10)
print("Next step: Open your 'finger_object_select.py' script and update the 'MODEL_PATH' variable to this new path.")
