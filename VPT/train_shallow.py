import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.set_seed import set_seed
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from models.vit_model import ViTForFlowerRecognition
from utils.data_loader import get_dataloaders
from utils.losses import CombinedLoss
from utils.augmentations import mixup_data


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if use_mixup:
            mixed_inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, alpha=0.2, use_cuda=device.type == "cuda"
            )

            logits, embeddings = model(mixed_inputs)

            loss_a = criterion(logits, embeddings, targets_a)
            loss_b = criterion(logits, embeddings, targets_b)
            loss = lam * loss_a + (1 - lam) * loss_b

            _, predicted = torch.max(logits.data, 1)
            correct += (
                lam * predicted.eq(targets_a.data).sum().float()
                + (1 - lam) * predicted.eq(targets_b.data).sum().float()
            )

        else:
            logits, embeddings = model(inputs)
            loss = criterion(logits, embeddings, targets)

            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == targets).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)

        progress_bar.set_postfix({"loss": loss.item()})

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

            logits, embeddings = model(inputs)
            loss = criterion(logits, embeddings, targets)

            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(epochs=50, batch_size=32, device="cpu"):
    print(f"Initializing VPT-Shallow training on {device}...")

    set_seed()

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = ViTForFlowerRecognition(num_classes=102, num_prompts=10).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    criterion = CombinedLoss(alpha=0.2).to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print("Starting VPT-Shallow Training Loop...")
    for epoch in range(epochs):
        start_time = time.time()

        use_mixup = True if epoch < (epochs - 10) else False

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=use_mixup
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        end_time = time.time()

        print(f"Epoch {epoch + 1}/{epochs} | Time: {end_time - start_time:.0f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> New best validation accuracy: {best_val_acc:.4f}. Saving checkpoint.")
            torch.save(model.state_dict(), "results/best_vit_vpt_model.pth")

    print("VPT-Shallow Training Complete. Best Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(epochs=50, batch_size=32, device=device)
