"""
Encapsulates the loading logic and auxiliary functions for the COCO dataset.
"""
import os
import random
import torchvision
from torchvision import transforms

class CocoDatasetWrapper:
    """
    A simple wrapper class for managing COCO Dataset objects.
    """
    def __init__(self, data_dir, split='train', transform=None):
        """
        :param data_dir: Root directory of COCO dataset, assumed to contain annotations/ and train2017/ val2017/ folders
        :param split: 'train' or 'val'
        :param transform: torchvision.transforms
        """
        if split not in ['train', 'val']:
            raise ValueError("split parameter should be 'train' or 'val'.")

        if split == 'train':
            ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
            img_dir = os.path.join(data_dir, "images", "train2017")
        else:  # val
            ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
            img_dir = os.path.join(data_dir, "images", "val2017")

        if transform is None:
            # If no transform is provided, use a default one
            transform = transforms.ToTensor()

        # Create CocoDetection dataset
        self.dataset = torchvision.datasets.CocoDetection(
            root=img_dir,
            annFile=ann_file,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def check_dataset_integrity(dataset, num_samples=100):
    """
    Simple check for dataset integrity by randomly reading several images
    to detect corruption or path errors.
    :param dataset: CocoDatasetWrapper or direct torch Dataset
    :param num_samples: Number of samples to randomly check
    """
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i in indices:
        try:
            img, target = dataset[i]
        except Exception as e:
            print(f"[Error] Unable to load at index {i}: {e}")
    print("[Info] Dataset integrity check completed.")

def get_coco_dataset(data_dir, split='train', transform=None):
    """
    Returns a CocoDatasetWrapper instance, or can directly return torchvision.datasets.CocoDetection.
    """
    return CocoDatasetWrapper(data_dir, split, transform)