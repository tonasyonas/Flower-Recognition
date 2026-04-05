# train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from models.resnet_baseline import ResNet18Baseline
from utils.data_loader import get_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Standard forward pass
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        
        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == targets).sum().item()

        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            running_loss += loss.item() * inputs.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_baseline_model(epochs=50, batch_size=32, device='cpu'):
    print(f"Initializing baseline training on {device}...")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    # 2. Init Model
    model = ResNet18Baseline(num_classes=102).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    # 3. Setup standard CE Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    # 4. Main Training Loop
    print("Starting Baseline Training Loop...")
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        end_time = time.time()
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {end_time - start_time:.0f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> New best validation accuracy: {best_val_acc:.4f}. Saving checkpoint.")
            torch.save(model.state_dict(), 'best_resnet_baseline.pth')
            
    print("Baseline Training Complete. Best Validation Accuracy:", best_val_acc)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_baseline_model(epochs=50, batch_size=32, device=device)
