# CNN Variant Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 CNN variants (dilated, deformable, depthwise-separable, hybrid) alongside the existing ResNet18 baseline to compare advanced convolution techniques on Oxford Flowers 102.

**Architecture:** All variants modify ResNet18's convolutional layers while keeping the same training pipeline (AdamW, lr=1e-3, CosineAnnealing, 50 epochs, CE loss). Each variant is a separate model class in `models/`. A unified training script handles all variants via CLI argument.

**Tech Stack:** PyTorch, torchvision (ResNet18 weights, `torchvision.ops.deform_conv2d`), tqdm, scikit-learn (evaluation)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `models/resnet_baseline.py` | Existing. Standard ResNet18 baseline. |
| `models/resnet_dilated.py` | **Create.** ResNet18 with dilated convolutions in layer3/layer4. |
| `models/resnet_deformable.py` | **Create.** ResNet18 with deformable convolutions in layer3/layer4. |
| `models/resnet_depthwise.py` | **Create.** ResNet18 with depthwise-separable convolutions (reduced params). |
| `models/resnet_hybrid.py` | **Create.** ResNet18 with dilated conv in layer3 + deformable conv in layer4 (novel). |
| `models/__init__.py` | **Modify.** Export all model classes. |
| `train_cnn.py` | **Create.** Unified training script for all CNN variants with CLI args. |
| `evaluate_cnn.py` | **Create.** Unified evaluation script: accuracy, classification report, confusion matrix for all CNN models. |
| `utils/data_loader.py` | **Modify.** Add `get_fewshot_dataloaders(k_shot)` for few-shot experiments. |

---

### Task 1: Dilated Convolution ResNet18

**Files:**
- Create: `models/resnet_dilated.py`

ResNet18 has 4 layers, each with 2 BasicBlocks. Each BasicBlock has two 3x3 convolutions. We replace the 3x3 convolutions in **layer3** and **layer4** with dilated convolutions (dilation=2, padding=2 to maintain spatial dimensions).

- [ ] **Step 1: Create the dilated ResNet18 model**

```python
# models/resnet_dilated.py
import torch.nn as nn
import torchvision.models as models

class ResNet18Dilated(nn.Module):
    """
    ResNet18 with dilated convolutions (dilation=2) in layer3 and layer4.
    Increases receptive field without adding parameters.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Replace convolutions in layer3 and layer4 with dilation=2
        self._apply_dilation(self.backbone.layer3, dilation=2)
        self._apply_dilation(self.backbone.layer4, dilation=2)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_dilation(self, layer, dilation):
        """Replace 3x3 convolutions in a ResNet layer with dilated versions."""
        for block in layer:
            # conv1: 3x3 conv in BasicBlock
            if isinstance(block.conv1, nn.Conv2d) and block.conv1.kernel_size == (3, 3):
                block.conv1 = nn.Conv2d(
                    block.conv1.in_channels, block.conv1.out_channels,
                    kernel_size=3, stride=block.conv1.stride[0],
                    padding=dilation, dilation=dilation, bias=False
                )
            # conv2: 3x3 conv in BasicBlock
            if isinstance(block.conv2, nn.Conv2d) and block.conv2.kernel_size == (3, 3):
                block.conv2 = nn.Conv2d(
                    block.conv2.in_channels, block.conv2.out_channels,
                    kernel_size=3, stride=1,
                    padding=dilation, dilation=dilation, bias=False
                )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
```

- [ ] **Step 2: Verify model builds and produces correct output shape**

```bash
cd D:/code/Flower-Recognition
python -c "
import torch
from models.resnet_dilated import ResNet18Dilated
model = ResNet18Dilated(num_classes=102)
x = torch.randn(2, 3, 224, 224)
logits, _ = model(x)
print('Output shape:', logits.shape)  # Expected: torch.Size([2, 102])
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable params:', params)
"
```
Expected: Output shape `[2, 102]`, param count ~11.2M (same as baseline since dilation doesn't add params).

- [ ] **Step 3: Commit**

```bash
git add models/resnet_dilated.py
git commit -m "feat: add ResNet18 with dilated convolutions in layer3/layer4"
```

---

### Task 2: Deformable Convolution ResNet18

**Files:**
- Create: `models/resnet_deformable.py`

Deformable convolution learns per-pixel offsets to shift the sampling grid. We need a wrapper module that:
1. Takes input features
2. Predicts offsets via a small conv layer
3. Applies `torchvision.ops.deform_conv2d` with those offsets

We replace the **first 3x3 conv** in each BasicBlock of **layer3** and **layer4**.

- [ ] **Step 1: Create the DeformableConv2d wrapper module**

```python
# models/resnet_deformable.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    """
    Wrapper around torchvision.ops.deform_conv2d.
    Learns per-pixel offsets to adapt the convolution sampling grid.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding

        # Offset prediction: 2 offsets (x, y) per kernel position
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Main convolution weight
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        offset = self.offset_conv(x)
        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=(self.stride, self.stride),
                             padding=(self.padding, self.padding))
```

- [ ] **Step 2: Create the ResNet18Deformable model**

```python
class ResNet18Deformable(nn.Module):
    """
    ResNet18 with deformable convolutions replacing the first 3x3 conv
    in each BasicBlock of layer3 and layer4.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Replace conv1 in each block of layer3 and layer4
        self._apply_deformable(self.backbone.layer3)
        self._apply_deformable(self.backbone.layer4)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_deformable(self, layer):
        """Replace conv1 in each BasicBlock with DeformableConv2d."""
        for block in layer:
            old_conv = block.conv1
            block.conv1 = DeformableConv2d(
                old_conv.in_channels, old_conv.out_channels,
                kernel_size=3, stride=old_conv.stride[0], padding=1
            )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
```

- [ ] **Step 3: Verify model builds and produces correct output shape**

```bash
python -c "
import torch
from models.resnet_deformable import ResNet18Deformable
model = ResNet18Deformable(num_classes=102)
x = torch.randn(2, 3, 224, 224)
logits, _ = model(x)
print('Output shape:', logits.shape)  # Expected: torch.Size([2, 102])
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable params:', params)
"
```
Expected: Output shape `[2, 102]`, param count slightly higher than baseline (~11.3M due to offset convs).

- [ ] **Step 4: Commit**

```bash
git add models/resnet_deformable.py
git commit -m "feat: add ResNet18 with deformable convolutions in layer3/layer4"
```

---

### Task 3: Depthwise-Separable Convolution ResNet18 (Parameter Reduction)

**Files:**
- Create: `models/resnet_depthwise.py`

Depthwise-separable convolution splits a standard conv into:
1. **Depthwise conv:** One filter per input channel (groups=in_channels)
2. **Pointwise conv:** 1x1 conv to mix channels

This reduces parameters from `C_in * C_out * K * K` to `C_in * K * K + C_in * C_out`.

We replace ALL 3x3 convolutions in layer2, layer3, and layer4.

- [ ] **Step 1: Create the DepthwiseSeparableConv module and model**

```python
# models/resnet_depthwise.py
import torch.nn as nn
import torchvision.models as models

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable convolution: depthwise (per-channel) + pointwise (1x1).
    Reduces parameters by ~8-9x compared to standard 3x3 conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResNet18Depthwise(nn.Module):
    """
    ResNet18 with depthwise-separable convolutions in layer2/layer3/layer4.
    Significantly reduces trainable parameters while maintaining structure.
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Replace 3x3 convs in layer2, layer3, layer4 with depthwise-separable
        self._apply_depthwise(self.backbone.layer2)
        self._apply_depthwise(self.backbone.layer3)
        self._apply_depthwise(self.backbone.layer4)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_depthwise(self, layer):
        """Replace 3x3 convolutions with depthwise-separable versions."""
        for block in layer:
            if isinstance(block.conv1, nn.Conv2d) and block.conv1.kernel_size == (3, 3):
                block.conv1 = DepthwiseSeparableConv(
                    block.conv1.in_channels, block.conv1.out_channels,
                    kernel_size=3, stride=block.conv1.stride[0], padding=1
                )
            if isinstance(block.conv2, nn.Conv2d) and block.conv2.kernel_size == (3, 3):
                block.conv2 = DepthwiseSeparableConv(
                    block.conv2.in_channels, block.conv2.out_channels,
                    kernel_size=3, stride=1, padding=1
                )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
```

- [ ] **Step 2: Verify model builds and check parameter reduction**

```bash
python -c "
import torch
from models.resnet_depthwise import ResNet18Depthwise
from models.resnet_baseline import ResNet18Baseline
model_dw = ResNet18Depthwise(num_classes=102)
model_bl = ResNet18Baseline(num_classes=102)
x = torch.randn(2, 3, 224, 224)
logits, _ = model_dw(x)
print('Output shape:', logits.shape)
params_dw = sum(p.numel() for p in model_dw.parameters() if p.requires_grad)
params_bl = sum(p.numel() for p in model_bl.parameters() if p.requires_grad)
print(f'Depthwise params: {params_dw:,} | Baseline params: {params_bl:,} | Reduction: {(1 - params_dw/params_bl)*100:.1f}%')
"
```
Expected: Output shape `[2, 102]`, significant parameter reduction (~30-50% fewer params).

- [ ] **Step 3: Commit**

```bash
git add models/resnet_depthwise.py
git commit -m "feat: add ResNet18 with depthwise-separable convolutions for parameter reduction"
```

---

### Task 4: Hybrid Dilated + Deformable ResNet18 (Novel Contribution)

**Files:**
- Create: `models/resnet_hybrid.py`

This is the **novel contribution**: combine dilated convolutions in layer3 (wider receptive field) with deformable convolutions in layer4 (adaptive spatial sampling). The intuition is that lower layers benefit from structured wider context, while higher layers benefit from input-dependent spatial adaptation.

- [ ] **Step 1: Create the hybrid model**

```python
# models/resnet_hybrid.py
import torch
import torch.nn as nn
import torchvision.models as models
from models.resnet_deformable import DeformableConv2d

class ResNet18Hybrid(nn.Module):
    """
    Novel hybrid: dilated convolutions in layer3 + deformable convolutions in layer4.
    Rationale: layer3 benefits from wider receptive field (dilation),
    layer4 benefits from input-adaptive spatial sampling (deformable).
    """
    def __init__(self, num_classes=102):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Layer3: dilated convolutions for wider receptive field
        self._apply_dilation(self.backbone.layer3, dilation=2)

        # Layer4: deformable convolutions for adaptive sampling
        self._apply_deformable(self.backbone.layer4)

        # Replace classifier head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def _apply_dilation(self, layer, dilation):
        """Replace 3x3 convolutions with dilated versions."""
        for block in layer:
            if isinstance(block.conv1, nn.Conv2d) and block.conv1.kernel_size == (3, 3):
                block.conv1 = nn.Conv2d(
                    block.conv1.in_channels, block.conv1.out_channels,
                    kernel_size=3, stride=block.conv1.stride[0],
                    padding=dilation, dilation=dilation, bias=False
                )
            if isinstance(block.conv2, nn.Conv2d) and block.conv2.kernel_size == (3, 3):
                block.conv2 = nn.Conv2d(
                    block.conv2.in_channels, block.conv2.out_channels,
                    kernel_size=3, stride=1,
                    padding=dilation, dilation=dilation, bias=False
                )

    def _apply_deformable(self, layer):
        """Replace conv1 in each BasicBlock with DeformableConv2d."""
        for block in layer:
            old_conv = block.conv1
            block.conv1 = DeformableConv2d(
                old_conv.in_channels, old_conv.out_channels,
                kernel_size=3, stride=old_conv.stride[0], padding=1
            )

    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
```

- [ ] **Step 2: Verify model builds and produces correct output shape**

```bash
python -c "
import torch
from models.resnet_hybrid import ResNet18Hybrid
model = ResNet18Hybrid(num_classes=102)
x = torch.randn(2, 3, 224, 224)
logits, _ = model(x)
print('Output shape:', logits.shape)  # Expected: torch.Size([2, 102])
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable params:', params)
"
```
Expected: Output shape `[2, 102]`.

- [ ] **Step 3: Commit**

```bash
git add models/resnet_hybrid.py
git commit -m "feat: add novel hybrid ResNet18 with dilated (layer3) + deformable (layer4)"
```

---

### Task 5: Update models/__init__.py

**Files:**
- Modify: `models/__init__.py`

- [ ] **Step 1: Export all model classes**

```python
# models/__init__.py
from .resnet_baseline import ResNet18Baseline
from .resnet_dilated import ResNet18Dilated
from .resnet_deformable import ResNet18Deformable
from .resnet_depthwise import ResNet18Depthwise
from .resnet_hybrid import ResNet18Hybrid

MODEL_REGISTRY = {
    'baseline': ResNet18Baseline,
    'dilated': ResNet18Dilated,
    'deformable': ResNet18Deformable,
    'depthwise': ResNet18Depthwise,
    'hybrid': ResNet18Hybrid,
}
```

- [ ] **Step 2: Commit**

```bash
git add models/__init__.py
git commit -m "feat: add model registry for all CNN variants"
```

---

### Task 6: Unified CNN Training Script

**Files:**
- Create: `train_cnn.py`

Single script that trains any CNN variant via `--model` flag. Identical hyperparameters across all variants for fair comparison. Logs training/validation loss and accuracy per epoch to a CSV file for plotting.

- [ ] **Step 1: Create the unified training script**

```python
# train_cnn.py
import argparse
import csv
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import MODEL_REGISTRY
from utils.data_loader import get_dataloaders
from utils.set_seed import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """Validate model, return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train CNN variants on Flowers 102")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=69)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model} on {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size)

    # Model
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(num_classes=102).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Logging
    os.makedirs('results', exist_ok=True)
    log_path = f'results/{args.model}_training_log.csv'
    checkpoint_path = f'results/best_{args.model}.pth'

    best_val_acc = 0.0

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

        for epoch in range(args.epochs):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            elapsed = time.time() - start

            writer.writerow([epoch + 1, f'{train_loss:.4f}', f'{train_acc:.4f}',
                             f'{val_loss:.4f}', f'{val_acc:.4f}', f'{lr:.6f}'])
            f.flush()

            print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> New best: {best_val_acc:.4f}")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {log_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify script runs with --help**

```bash
python train_cnn.py --help
```
Expected: Shows argument options with model choices.

- [ ] **Step 3: Commit**

```bash
git add train_cnn.py
git commit -m "feat: add unified CNN training script with CLI args and CSV logging"
```

---

### Task 7: CNN Evaluation Script

**Files:**
- Create: `evaluate_cnn.py`

Loads a trained checkpoint, runs on test set, outputs accuracy, classification report, and confusion matrix.

- [ ] **Step 1: Create the evaluation script**

```python
# evaluate_cnn.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from models import MODEL_REGISTRY
from utils.data_loader import get_dataloaders


def evaluate(model, dataloader, device):
    """Run inference and return all predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN model on Flowers 102 test set")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(num_classes=102).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded {args.model} from {args.checkpoint}")

    # Load test data
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Evaluate
    preds, labels = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Classification report
    report = classification_report(labels, preds)
    report_path = f'results/{args.model}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Model: {args.model}\nTest Accuracy: {acc:.4f}\n\n{report}")
    print(f"Report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{args.model} - Confusion Matrix (Acc: {acc:.4f})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = f'results/{args.model}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=200)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add evaluate_cnn.py
git commit -m "feat: add unified CNN evaluation script with metrics and confusion matrix"
```

---

### Task 8: Few-Shot Data Loader

**Files:**
- Modify: `utils/data_loader.py`

Add a function to subsample k images per class from the training set for few-shot experiments.

- [ ] **Step 1: Add get_fewshot_dataloaders function**

Add this function to `utils/data_loader.py`:

```python
def get_fewshot_dataloaders(k_shot, batch_size=32, data_dir='./data'):
    """
    Load Flowers 102 with only k_shot training images per class.
    Validation and test sets remain unchanged.
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.Flowers102(root=data_dir, split='train', download=True, transform=transform_train)
    val_dataset = datasets.Flowers102(root=data_dir, split='val', download=True, transform=transform_test)
    test_dataset = datasets.Flowers102(root=data_dir, split='test', download=True, transform=transform_test)

    # Subsample k images per class
    targets = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    classes = targets.unique()
    indices = []
    for c in classes:
        class_indices = (targets == c).nonzero(as_tuple=True)[0].tolist()
        indices.extend(class_indices[:k_shot])

    train_subset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
```

- [ ] **Step 2: Add --k_shot argument to train_cnn.py**

In `train_cnn.py`, add the argument and modify the data loading:

```python
# In argparse section:
parser.add_argument('--k_shot', type=int, default=None,
                    help='Number of training images per class (few-shot). Default: use full training set')

# In data loading section, replace get_dataloaders call:
if args.k_shot:
    from utils.data_loader import get_fewshot_dataloaders
    train_loader, val_loader, _ = get_fewshot_dataloaders(args.k_shot, batch_size=args.batch_size)
    print(f"Few-shot mode: {args.k_shot} images per class")
else:
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size)
```

- [ ] **Step 3: Verify few-shot loader produces correct subset size**

```bash
python -c "
from utils.data_loader import get_fewshot_dataloaders
train_loader, _, _ = get_fewshot_dataloaders(k_shot=2)
total = sum(len(b[0]) for b in train_loader)
print(f'Training samples with k=2: {total}')  # Expected: 204 (2 * 102 classes)
"
```

- [ ] **Step 4: Commit**

```bash
git add utils/data_loader.py train_cnn.py
git commit -m "feat: add few-shot data loader and --k_shot CLI argument"
```

---

## Execution Commands Summary

After all tasks are complete, train all variants:

```bash
# Full training set (10-shot, the default)
python train_cnn.py --model baseline
python train_cnn.py --model dilated
python train_cnn.py --model deformable
python train_cnn.py --model depthwise
python train_cnn.py --model hybrid

# Few-shot experiments (run for each model)
python train_cnn.py --model baseline --k_shot 1 --epochs 30
python train_cnn.py --model baseline --k_shot 2 --epochs 30
python train_cnn.py --model baseline --k_shot 5 --epochs 30
# ... repeat for each model variant

# Evaluation
python evaluate_cnn.py --model baseline --checkpoint results/best_baseline.pth
python evaluate_cnn.py --model dilated --checkpoint results/best_dilated.pth
# ... repeat for each model
```

## Expected Results Table (for report)

| Model | Params | 1-shot | 2-shot | 5-shot | 10-shot |
|-------|--------|--------|--------|--------|---------|
| ResNet18 Baseline | ~11.2M | - | - | - | - |
| ResNet18 Dilated | ~11.2M | - | - | - | - |
| ResNet18 Deformable | ~11.3M | - | - | - | - |
| ResNet18 Depthwise | ~7-8M | - | - | - | - |
| ResNet18 Hybrid (Novel) | ~11.3M | - | - | - | - |
