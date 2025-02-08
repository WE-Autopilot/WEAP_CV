import pandas as pd
import yaml 
import os
import cv2

def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data

# Example usage
dataset_info = load_yaml("dataset/dataset.yaml")

# Print dataset information
print(f"Train Path: {dataset_info['train']}")
print(f"Validation Path: {dataset_info['val']}")
print(f"Test Path: {dataset_info['test']}")
print(f"Number of Classes: {dataset_info['nc']}")
print("Class Names:", dataset_info["names"])



