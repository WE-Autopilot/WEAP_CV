import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np

'''
Define a small CNN.
We want 8 layers - conv1, batch + poo1, relu1, conv2, batch + pool2, relu2, fc1, fc2  
Split up the two parts 1.Model 2.Classifier
'''
class SmallCNN(nn.Module):  
    def __init__(self, num_classes=2, input_size=128) -> None:
        super().__init__()
        
        # calculate size after convolution and pooling
        feature_size = ((input_size - 2) // 2 - 2) // 2
        # Calculate the flattened feature size
        self.flat_features = 64 * feature_size * feature_size

        # Small CNN part
        self.stop_sign_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0), # 64 features/outputs, input channels = 3 (RGB), square kernel of size 3, 
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            # Dropout for regularization to avoid overfitting
            nn.Dropout2d(0.20)
        )

        # Fully connected part
        self.stop_sign_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        def forward(self, x):
            x = self.stop_sign_cnn(x)
            x = self.stop_sign_classifier(x)
            return x

# Create the model
detection_model = SmallCNN()

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detection_model.to(device)

# define a loss function and optimizer
loss_fcn = nn.CrossEntropyLoss()
optimizer = optim.Adam(detection_model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-4) # weigh decay for regularization

# learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

def train_model(model, train_loader, val_loader, num_epochs = 25, ):
    best = 0.0 # holds best accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        running_loss = 0.0

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            print('.', end='', flush=True)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # outputs is a tensor of shape [batch_size,2] and torch.max igores the first value
            loss = loss_fcn(outputs, labels)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            # loss and corrects
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') 

    # Validation phase
    model.eval()  # Set model to evaluate mode
    val_loss = 0.0
    val_corrects = 0
        
    # No gradient calculation needed for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fcn(outputs, labels)
                
            # Statistics
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {val_acc:.4f}')
    
    print(f'Best validation accuracy: {best_acc:.4f}')
    return model




        
