"""
Contains a simple training process example that can be modified based on project requirements.
"""
import torch
import torch.nn as nn
import torch.optim as optim

def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    """
    Example of training process for one epoch
    :param model: nn.Module, your model
    :param dataloader: DataLoader for training set
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: 'cpu' or 'cuda'
    :return: average loss for this epoch
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        dummy_labels = []
        for ann_list in targets:
            if len(ann_list) > 0:
                # Use category_id from first annotation for demo
                dummy_labels.append(ann_list[0]['category_id'])
            else:
                # If no annotation in image, use 0
                dummy_labels.append(0)
        dummy_labels = torch.tensor(dummy_labels, dtype=torch.long, device=device)

        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, dummy_labels)
        
        # Backward pass + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device='cpu'):
    """
    A simple validation/testing process example
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # Same processing for targets -> dummy_labels
            dummy_labels = []
            for ann_list in targets:
                if len(ann_list) > 0:
                    dummy_labels.append(ann_list[0]['category_id'])
                else:
                    dummy_labels.append(0)
            dummy_labels = torch.tensor(dummy_labels, dtype=torch.long, device=device)
            outputs = model(images)
            loss = criterion(outputs, dummy_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=5, 
    lr=1e-3, 
    device='cpu'
):
    """
    Handles the entire training process
    """
    # Assume classification task with num_classes output size
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    print("Training completed.")
    return model