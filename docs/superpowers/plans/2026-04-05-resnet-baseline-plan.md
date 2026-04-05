# ResNet18 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standard pre-trained ResNet18 baseline model, a training script without advanced augmentations, and an evaluation notebook to compare against the ViT-VPT method for the Oxford Flowers 102 dataset.

**Architecture:** A standard ResNet18 model loaded with `torchvision` pre-trained weights. The final fully connected layer will be replaced with a new linear layer for 102 classes. Standard data loading and cross-entropy loss will be used.

**Tech Stack:** PyTorch, Torchvision, Scikit-Learn, Jupyter

---

### Task 1: Create the ResNet18 Model Wrapper

**Files:**
- Create: `models/resnet_baseline.py`

- [ ] **Step 1: Write the model implementation**
Create a wrapper that loads the pretrained ResNet18 and modifies the final layer.

```python
# models/resnet_baseline.py
import torch.nn as nn
import torchvision.models as models

class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=102):
        super(ResNet18Baseline, self).__init__()
        # Load pre-trained ResNet18
        # Using the standard weights (ImageNet)
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # We only need the logits for the baseline (no embeddings needed for standard CE loss)
        logits = self.backbone(x)
        return logits, None  # Returning None for embeddings to match ViT tuple unpacking in eval if needed
```

- [ ] **Step 2: Commit the model**
```bash
git add models/resnet_baseline.py
git commit -m "feat: add resnet18 baseline model"
```

### Task 2: Create the Baseline Training Script

**Files:**
- Create: `train_baseline.py`

- [ ] **Step 1: Write the training script logic**
This script mirrors `train.py` but removes MixUp and Triplet Loss.

```python
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
```

- [ ] **Step 2: Commit the training script**
```bash
git add train_baseline.py
git commit -m "feat: add resnet18 baseline training script"
```

### Task 3: Create the Baseline Evaluation Script

**Files:**
- Create: `evaluation_baseline.py`

- [ ] **Step 1: Write the evaluation script**
This will be a python script that loads the trained weights and dumps the confusion matrix and report.

```python
# evaluation_baseline.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from models.resnet_baseline import ResNet18Baseline
from utils.data_loader import get_dataloaders

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dataloaders
    _, _, test_loader = get_dataloaders(batch_size=32)

    # Initialize model
    model = ResNet18Baseline(num_classes=102)

    # Load the best checkpoint
    checkpoint_path = 'best_resnet_baseline.pth'
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint from '{checkpoint_path}'")
    except FileNotFoundError:
        print(f"Error: '{checkpoint_path}' not found. Make sure you have trained the model first.")
        exit(1)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating Baseline"):
            inputs, targets = inputs.to(device), targets.to(device)

            logits, _ = model(inputs)
            _, predicted = torch.max(logits.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_accuracy = np.mean(all_preds == all_targets)
    print(f"Baseline Test Accuracy: {test_accuracy * 100:.2f}%")

    report = classification_report(all_targets, all_preds, zero_division=0)
    
    with open("classification_report_baseline.txt", "w") as f:
        f.write(report)
    print("Report saved to classification_report_baseline.txt")

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap="Reds", cbar=True)
    plt.title("Baseline Confusion Matrix (102 Classes)", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.tight_layout()
    plt.savefig("confusion_matrix_baseline.png", dpi=300)
```

- [ ] **Step 2: Commit the evaluation script**
```bash
git add evaluation_baseline.py
git commit -m "feat: add resnet18 baseline evaluation script"
```
