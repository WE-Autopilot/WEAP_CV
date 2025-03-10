import torch
from torch import nn
from torch import optim as optim
from torchvision import datasets


#Questions
#- Where to input batch norm and dropout, it seems inconclusive and confusing

#=====================================
#CONSTANTS
#=====================================
NUM_EPOCHS = 2


#=====================================
#Data Import and Cleaning
#=====================================


#=====================================
#MODEL DEFINITION
#=====================================

# Optimized CNN Model for Real-Time Stop Sign Classification

# Model Architecture:
# 1. Conv2D (3x3, 64 filters, stride=1, padding=1)  -> Output: 128x128x64
# 2. MaxPool2D (2x2, stride=2)                      -> Output: 64x64x64
# 3. ReLU Activation
# 4. Conv2D (2x2, 32 filters, stride=1, padding=1)  -> Output: 64x64x32
# 5. MaxPool2D (2x2, stride=2)                      -> Output: 32x32x32
# 6. ReLU Activation
# 7. Global Average Pooling                         -> Output: 32
# 8. Fully Connected (FC2: 32 -> 2)                 -> Output: 2 (Binary Classification)
#

# Optimization Strategies:
# - Used Global Average Pooling (GAP) instead of Flatten + Large FC layer.
# - Reduced the number of parameters significantly.
# - Maintains accuracy while reducing computational cost for real-time performance.

class CNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(kernel_size = 3, stride = 1, padding = "same", in_channels = 3, out_channels = 64),
            nn.MaxPool2d(stride= 2, kernel_size= 2),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, stride = 1, padding = "same", in_channels = 64, out_channels = 32),
            nn.MaxPool2d(stride= 2, kernel_size= 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2)
            #input sigmoid
        )

    def forward(self, x):
        return self.model(x)
    
    def train_model():

        loss_func = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(NUM_EPOCHS):

            running_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):

                #PLACEHOLDER FOR INPUT
                inputs, label = data


    


    

    




    
