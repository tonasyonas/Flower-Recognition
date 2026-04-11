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
from utils.data_loader import get_dataloaders, get_fewshot_dataloaders
from utils.set_seed import set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional AMP, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits, _ = model(inputs)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
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
    parser.add_argument('--k_shot', type=int, default=None,
                        help='Few-shot: number of training images per class')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model} on {device}")

    # Data
    if args.k_shot:
        train_loader, val_loader, _ = get_fewshot_dataloaders(args.k_shot, batch_size=args.batch_size)
        print(f"Few-shot mode: {args.k_shot} images per class")
    else:
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

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Mixed precision (AMP) enabled")

    # Logging
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    shot_suffix = f'_{args.k_shot}shot' if args.k_shot else ''
    log_path = f'results/logs/{args.model}{shot_suffix}_training_log.csv'
    checkpoint_path = f'results/checkpoints/best_{args.model}{shot_suffix}.pth'

    best_val_acc = 0.0

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

        for epoch in range(args.epochs):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
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
