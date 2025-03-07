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
    img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.9, 1.2), beta=np.random.randint(-15, 15))  # Reduced contrast variation + Reduced brightness shift

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[..., 1] = img_hsv[..., 1] * np.random.uniform(0.85, 1.15)  
    img_hsv[..., 0] = img_hsv[..., 0] + np.random.randint(-5, 5)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    if np.random.rand() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)  

    if np.random.rand() < 0.15:
        kernel_size = 3
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        img = cv2.filter2D(img, -1, kernel_motion_blur)

    if np.random.rand() < 0.2:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(50, 90)]
        _, enc_img = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(enc_img, cv2.IMREAD_UNCHANGED)

    if np.random.rand() < 0.2:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return Image.fromarray(img)

def get_stop_sign_transforms():
    """
    Returns the transformation pipeline for dynamic augmentation during training.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_cv2_transforms(img)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

class StopSignDataset(Dataset):
    """
    Custom dataset class for Stop Sign detection that applies transformations during training.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        augmented_image = self.transform(image) if self.transform else transforms.ToTensor()(image)

        return augmented_image, label  # Only return augmented image and label
    
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

def create_train_val_test_loaders(image_paths, labels, batch_size=8, train_ratio=0.7, test_ratio=0.15, num_workers=4):
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