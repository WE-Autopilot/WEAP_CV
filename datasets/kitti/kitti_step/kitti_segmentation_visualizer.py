# Panoptic Map + Image -> Mask Visualizer
# Rough draft for a segmentation visualizer that takes in an image and panoptic image, then overlays them
# This will be needed and useful for verifying that image augmentation does not cause errors in the masks 

import cv2 as cv
import numpy as np
import sys

# Numpy array of colour constants used for mask matrix, uses BRG (bc of OpenCV), key values are set to the Semantic Labeling Convention Used in Kitti
LABEL_BGR = {
    0: np.array([128, 64, 128], dtype=np.uint8),   # road
    1: np.array([244, 35, 232], dtype=np.uint8),   # sidewalk
    2: np.array([70, 70, 70], dtype=np.uint8),     # building
    3: np.array([102, 102, 156], dtype=np.uint8),  # wall
    4: np.array([190, 153, 153], dtype=np.uint8),  # fence
    5: np.array([153, 153, 153], dtype=np.uint8),  # pole
    6: np.array([30, 170, 250], dtype=np.uint8),   # traffic light
    7: np.array([0, 220, 220], dtype=np.uint8),    # traffic sign
    8: np.array([35, 142, 107], dtype=np.uint8),   # vegetation
    9: np.array([152, 251, 152], dtype=np.uint8),  # terrain
    10: np.array([180, 130, 70], dtype=np.uint8),  # sky
    11: np.array([60, 20, 220], dtype=np.uint8),   # person
    12: np.array([0, 0, 255], dtype=np.uint8),     # rider
    13: np.array([142, 0, 0], dtype=np.uint8),     # car
    14: np.array([70, 0, 0], dtype=np.uint8),      # truck
    15: np.array([100, 60, 0], dtype=np.uint8),    # bus
    16: np.array([100, 80, 0], dtype=np.uint8),    # train
    17: np.array([230, 0, 0], dtype=np.uint8),     # motorcycle
    18: np.array([32, 11, 119], dtype=np.uint8),   # bicycle
    255: np.array([0, 0, 0], dtype=np.uint8)       # void
}

# Takes in path of image file and returns an openCV matrix
def file_loader(path):
    imgMatrix = cv.imread(path)
    if (np.count_nonzero(imgMatrix) == 0):
        raise RuntimeError("Image was unable to be read")
    return imgMatrix


# Extremely slow, works for now, but if the mask is generated in real time this needs to be optimized
# Takes in image matrix of panoptic map, returns mask
def generate_mask(imgMatrix):
    height, width, channels = np.shape(imgMatrix)
    maskMatrix = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixel = imgMatrix[i,j] #OpenCV uses BGR values
            maskMatrix[i,j] = LABEL_BGR.get(pixel[2], np.array([0, 0, 0], dtype=np.uint8))
    return maskMatrix

def create_overlay(imgMatrix, maskMatrix):
    alpha = 0.2
    segmentationMatrix = cv.addWeighted(maskMatrix, alpha, imgMatrix, 1 - alpha, 0)
    return segmentationMatrix

image_path = "C:/Users/Jairdan C/Desktop/WEAP/WEAP_CV/datasets/kitti/kitti_step/images/training/0000/000000.png"
panoptic_path = "C:/Users/Jairdan C/Desktop/WEAP/WEAP_CV/datasets/kitti/kitti_step/panoptic_maps/train/0000/000000.png"


imgMatrix = file_loader(image_path)
maskMatrix = file_loader(panoptic_path)
maskMatrix = generate_mask(maskMatrix)

segmentationMask = create_overlay(imgMatrix, maskMatrix)

cv.imshow("Display window", segmentationMask)
k = cv.waitKey(0)


