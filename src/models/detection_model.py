import torch
import torch.nn as nn

# we want 7 layers - conv1, pool1, relu1, conv2, pool2, relu2, fc1 (fully connected)
class SmallCNN(nn.Module):  
    def __init__(self) -> None:
        super().__init__(self)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d()
        self.stop_sign_detection = nn.Sequential(
            # we want 3 channels because our input size is 128x128x3
            # 64 features, square kernel of size 3
            nn.functional.conv2d(3, 64, 3, 3),
            nn.pool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.functional.Conv2d(3, 32, 3, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Linear(9216, 32),
            nn.Linear(32,2)
        )

# detection_model = SmallCNN()
