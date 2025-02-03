"""
dataset_loader.py - Handles dataset setup (Downloading, Preprocessing)

in these librarires we will be handling the dataset setup, downloading, and preprocessing 
"""

import os

def setup_datasets():
    """Initialize dataset directory structure"""
    os.makedirs("datasets/kitti", exist_ok=True)
    os.makedirs("datasets/coco", exist_ok=True)
    print(" Dataset directories are set up.")

if __name__ == "__main__":
    setup_datasets()
