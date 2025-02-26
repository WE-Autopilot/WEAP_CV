import cv2 as cv
import numpy as np

# Dictionary mapping KITTI semantic labels to their corresponding BGR color values.
# Note: OpenCV uses BGR format; the label index is assumed to be encoded in the red channel.
KITTI_LABEL_COLORS = {
    0: np.array([128, 64, 128], dtype=np.uint8),   # road
    1: np.array([244, 35, 232], dtype=np.uint8),   # sidewalk
    2: np.array([70, 70, 70], dtype=np.uint8),     # building
    3: np.array([102, 102, 156], dtype=np.uint8),   # wall
    4: np.array([190, 153, 153], dtype=np.uint8),   # fence
    5: np.array([153, 153, 153], dtype=np.uint8),   # pole
    6: np.array([30, 170, 250], dtype=np.uint8),    # traffic light
    7: np.array([0, 220, 220], dtype=np.uint8),     # traffic sign
    8: np.array([35, 142, 107], dtype=np.uint8),    # vegetation
    9: np.array([152, 251, 152], dtype=np.uint8),   # terrain
    10: np.array([180, 130, 70], dtype=np.uint8),   # sky
    11: np.array([60, 20, 220], dtype=np.uint8),    # person
    12: np.array([0, 0, 255], dtype=np.uint8),      # rider
    13: np.array([142, 0, 0], dtype=np.uint8),      # car
    14: np.array([70, 0, 0], dtype=np.uint8),       # truck
    15: np.array([100, 60, 0], dtype=np.uint8),     # bus
    16: np.array([100, 80, 0], dtype=np.uint8),     # train
    17: np.array([230, 0, 0], dtype=np.uint8),      # motorcycle
    18: np.array([32, 11, 119], dtype=np.uint8),    # bicycle
    255: np.array([0, 0, 0], dtype=np.uint8)        # void
}

def load_image(path):
    
    #Loads an image from the given file path using OpenCV.
    #Raises a RuntimeError if the image cannot be read.
    
    image = cv.imread(path)
    if image is None or np.count_nonzero(image) == 0:
        raise RuntimeError("Image at path '{}' was unable to be read".format(path))
    return image

def generate_mask(panoptic_image):
    
    #Generates a segmentation mask from the panoptic image.
    #For each pixel in the panoptic image, the red channel value (index 2) is used as a key
    #to retrieve the corresponding BGR color from KITTI_LABEL_COLORS.
    #If the label is not found, it defaults to black.
    #Note: This implementation is slow and may need optimization for real-time applications.

    height, width, channels = panoptic_image.shape
    mask = np.zeros((height, width, channels), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            pixel = panoptic_image[row, col]  # OpenCV uses BGR; label is in red channel
            mask[row, col] = KITTI_LABEL_COLORS.get(pixel[2], np.array([0, 0, 0], dtype=np.uint8))
    return mask

def create_overlay(image, mask):
    
    #Creates an overlay by blending the original image with the segmentation mask.
    #The blending factor (alpha) determines the transparency of the mask.

    alpha = 0.5
    overlay = cv.addWeighted(mask, alpha, image, 1 - alpha, 0)
    return overlay

# Absolute path to the input image and its corresponding panoptic map.
image_path = "C:/Users/Jairdan C/Desktop/WEAP/WEAP_CV/datasets/kitti/kitti_step/images/training/0000/000000.png"
panoptic_path = "C:/Users/Jairdan C/Desktop/WEAP/WEAP_CV/datasets/kitti/kitti_step/panoptic_maps/train/0000/000000.png"

# Load the original image and panoptic map.
image = load_image(image_path)
panoptic_image = load_image(panoptic_path)

# Generate the segmentation mask and create an overlay.
mask = generate_mask(panoptic_image)
segmentation_overlay = create_overlay(image, mask)

# Display the resulting overlay.
cv.imshow("Segmentation Overlay", segmentation_overlay)
cv.waitKey(0)
cv.destroyAllWindows()


