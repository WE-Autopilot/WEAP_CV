import torch
import torch.nn as nn
from datasets.coco import get_coco_dataset, check_dataset_integrity
from datasets.data_loader import create_train_val_test_loaders
from torchvision import transforms
from train import train_model
import datasets.data_loader as data_loader

def main():
    '''
    data_dir = "./coco"

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    coco_dataset = get_coco_dataset(data_dir, split='train', transform=transform)

    
    check_dataset_integrity(coco_dataset)


    train_loader, test_loader, val_loader = create_train_val_test_loaders(
        coco_dataset.dataset,  
        batch_size=4,
        train_ratio=0.7,
        test_ratio=0.15,
        num_workers=2
    )

    
    model = nn.Sequential(
        nn.Flatten(),         
        nn.Linear(3*224*224, 100), 
        nn.ReLU(),
        nn.Linear(100, 80)   
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=3, 
        lr=1e-3, 
        device=device
    )
    '''
    
    # testing the dataloader
    data_loader.download_roboflow_dataset('https://app.roboflow.com/ds/LXl5gthuky?key=9cEzxHzAiX')

if __name__ == '__main__':
    main()
