"""
baseline_model.py - Load and test a pre-trained YOLOv8 model

Running this code will load a YOLOv8 model and test it on a sample image named test.png.
"""

from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Smallest YOLOv8 model

# Test inference on a sample image
results = model("datasets/test.png")


