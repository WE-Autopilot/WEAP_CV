import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead
from torchvision.transforms.functional import to_tensor
import torch.optim as optim

# --- Model setup ---
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)

# Replace detection head for 10 classes (9 + 1 for background)
num_classes = 10
model.head = SSDHead(
    in_channels=[512, 1024, 512, 256, 256, 256],
    num_anchors=model.anchor_generator.num_anchors_per_location(),
    num_classes=num_classes
)

# Freeze backbone for fine-tuning
for param in model.backbone.parameters():
    param.requires_grad = False

# --- Transform for object detection ---
def detection_transform(image, target):
    return to_tensor(image), target  # Convert image to tensor, keep target unchanged

# --- Dataset ---
dataset = CocoDetection(
    root="bdd100k/images/train",
    annFile="bdd100k/labels/coco_train.json",
    transforms=detection_transform
)

# --- DataLoader with COCO collate function ---
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
)

# --- Optimizer for only trainable parameters ---
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005, momentum=0.9, weight_decay=0.0005
)

# --- Training loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(10):  # Example: 10 epochs
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} finished. Loss: {losses.item():.4f}")
