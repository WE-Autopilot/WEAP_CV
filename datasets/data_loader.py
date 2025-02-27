"""
Encapsulates DataLoader-related logic, including splitting and parallel loading and transformations.
"""
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset

def apply_cv2_transforms(img):
    """
    Applies OpenCV-based augmentations dynamically when images are loaded.
    :param img: PIL Image (converted to OpenCV format)
    :return: Transformed PIL Image
    """
    img = np.array(img)  # Convert PIL image to OpenCV format (NumPy array)

    # **Brightness & Contrast Adjustments**
    alpha = np.random.uniform(0.8, 1.5)  # Contrast factor
    beta = np.random.randint(-30, 30)    # Brightness offset
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # **Saturation & Hue Adjustments**
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV
    img_hsv[..., 1] = img_hsv[..., 1] * np.random.uniform(0.7, 1.3)  # Modify saturation
    img_hsv[..., 0] = img_hsv[..., 0] + np.random.randint(-10, 10)  # Modify hue
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)  # Convert back to RGB

    # **Gaussian Blur & Motion Blur**
    if np.random.rand() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Simulate blur

    if np.random.rand() < 0.2:
        kernel_size = 3
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        img = cv2.filter2D(img, -1, kernel_motion_blur)

    # **JPEG Compression & Gaussian Noise**
    if np.random.rand() < 0.3:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(30, 90)]
        _, enc_img = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(enc_img, cv2.IMREAD_UNCHANGED)

    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    # **Selective Red Boosting**
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[..., 1] = img_hsv[..., 1] * 1.1  # Slightly boost red saturation
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # **Weather Simulation (Fog & Shadows)**
    if np.random.rand() < 0.2:
        fog_intensity = np.random.uniform(0.3, 0.7)
        fog = np.full_like(img, 255, dtype=np.uint8)
        img = cv2.addWeighted(img, 1 - fog_intensity, fog, fog_intensity, 0)

    if np.random.rand() < 0.2:
        h, w, _ = img.shape
        shadow = np.random.uniform(0.3, 0.7, (h, w, 3)) * 255
        img = cv2.addWeighted(img, 1, shadow.astype(np.uint8), -0.5, 0)

    return Image.fromarray(img)  # Convert back to PIL format

# ========== APPLY TRANSFORMATION PIPELINE ==========
def get_stop_sign_transforms():
    """
    Returns the best transformation pipeline for dynamic augmentation during training.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_cv2_transforms(img)),  # Apply OpenCV-based augmentations dynamically

        # Torchvision Transformations
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# ========== MODIFIED DATASET CLASS ==========
class StopSignDataset(Dataset):
    """
    Custom dataset class for Stop Sign detection that returns both the original and augmented images.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # List of image file paths
        self.labels = labels  # Corresponding labels
        self.transform = transform  # Transformation pipeline

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Load image
        label = self.labels[idx]  # Get label

        original_image = transforms.ToTensor()(image)  # Convert original image to tensor

        augmented_image = self.transform(image) if self.transform else original_image  # Apply augmentation

        return original_image, augmented_image, label  # Return both images and label
    

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """
    Creates and returns a DataLoader
    :param dataset: torch.utils.data.Dataset
    :param batch_size: batch size for training
    :param shuffle: whether to shuffle the data order
    :param num_workers: number of worker processes
    :return: DataLoader
    """
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader

def split_dataset(dataset, train_ratio=0.7, test_ratio=0.15, seed=42):
    """
    Splits a dataset into train, test, and validation subsets.
    :param dataset: torch.utils.data.Dataset
    :param train_ratio: proportion of training set
    :param test_ratio: proportion of test set
    :param seed: random seed for reproducibility
    :return: (train_dataset, test_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = int(dataset_size * test_ratio)
    val_size = dataset_size - train_size - test_size

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    return random_split(dataset, [train_size, test_size, val_size])

def create_train_val_test_loaders(dataset, batch_size=8, train_ratio=0.7, test_ratio=0.15, num_workers=4):
    """
    Combines dataset splitting and DataLoader creation.
    :return: (train_loader, test_loader, val_loader)
    """
    dataset = StopSignDataset(image_paths=image_paths, labels=labels, transform=get_stop_sign_transforms())
    train_dataset, test_dataset, val_dataset = split_dataset(dataset, train_ratio, test_ratio)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = create_dataloader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader   = create_dataloader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, val_loader