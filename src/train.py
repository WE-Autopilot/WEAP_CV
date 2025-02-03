"""
train.py - Train a YOLOv8 model on a custom dataset

we need to config and complile a dataset for training.
This is just a skeleton setup of how it will be training a YOLOv8 model on a custom dataset.
"""

from ultralytics import YOLO

# Load YOLOv8 model configuration for training
model = YOLO("yolov8n.yaml")  # Using YOLOv8 Nano config

# Train on custom dataset
model.train(data="datasets/data.yaml", epochs=50, batch=16)

print("Training Complete")
