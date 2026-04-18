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
from utils.losses import batch_hard_triplet_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None,
                    triplet_alpha=0.0, triplet_margin=0.5):
    """Train one epoch with optional AMP. MixUp is handled in the dataloader collate_fn.
    Triplet loss is added to CE loss when triplet_alpha > 0 (requires model to return embeddings)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_triplet = triplet_alpha > 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits, embeddings = model(inputs)
            loss = criterion(logits, targets)
        if use_triplet and embeddings is not None and targets.ndim == 1:
            # Triplet requires hard labels; skip if MixUp produced soft labels.
            # Compute outside autocast in float32 for numerical stability.
            triplet = batch_hard_triplet_loss(embeddings.float(), targets, margin=triplet_margin)
            loss = loss + triplet_alpha * triplet

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(logits, 1)
        # Targets may be soft labels (batch, num_classes) from MixUp or hard labels (batch,)
        if targets.ndim == 1:
            correct += (predicted == targets).sum().item()
        else:
            correct += (predicted == targets.argmax(dim=1)).sum().item()
        total += inputs.size(0)

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
    parser = argparse.ArgumentParser(description="Train CNN/VPT variants on Flowers 102")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--k_shot', type=int, default=None,
                        help='Few-shot: number of training images per class')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='MixUp alpha (0=disabled, 0.2=recommended). Uses official torchvision v2.MixUp.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0=disabled, 0.1=recommended)')
    parser.add_argument('--strong_aug', action='store_true',
                        help='Use stronger augmentation (RandomResizedCrop, ColorJitter)')
    parser.add_argument('--num_prompts', type=int, default=10,
                        help='Number of visual prompts (for VPT models only)')
    parser.add_argument('--triplet_alpha', type=float, default=0.0,
                        help='Weight for batch-hard triplet loss (0=disabled, 0.2=recommended). '
                             'Total loss = CE + triplet_alpha * triplet.')
    parser.add_argument('--triplet_margin', type=float, default=0.5,
                        help='Margin for triplet loss')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model} on {device} (seed={args.seed})")

    # Data (MixUp applied via collate_fn in dataloader)
    if args.k_shot:
        train_loader, val_loader, _ = get_fewshot_dataloaders(
            args.k_shot, batch_size=args.batch_size, mixup_alpha=args.mixup,
            strong_aug=args.strong_aug)
        print(f"Few-shot mode: {args.k_shot} images per class")
    else:
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=args.batch_size, mixup_alpha=args.mixup,
            strong_aug=args.strong_aug)
    if args.strong_aug:
        print("Strong augmentation enabled (RandomResizedCrop, ColorJitter)")

    # Model (VPT models accept num_prompts kwarg)
    model_cls = MODEL_REGISTRY[args.model]
    is_vpt = args.model.startswith('vpt_')
    if is_vpt:
        model = model_cls(num_classes=102, num_prompts=args.num_prompts).to(device)
    else:
        model = model_cls(num_classes=102).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")
    if args.triplet_alpha > 0:
        print(f"Triplet loss enabled (alpha={args.triplet_alpha}, margin={args.triplet_margin})")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Mixed precision (AMP) enabled")
    if args.mixup > 0:
        print(f"MixUp enabled (alpha={args.mixup}, torchvision v2)")

    # Logging
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    shot_suffix = f'_{args.k_shot}shot' if args.k_shot else ''
    mixup_suffix = f'_mixup{args.mixup}' if args.mixup > 0 else ''
    ls_suffix = f'_ls{args.label_smoothing}' if args.label_smoothing > 0 else ''
    aug_suffix = '_strongaug' if args.strong_aug else ''
    prompt_suffix = f'_p{args.num_prompts}' if is_vpt and args.num_prompts != 10 else ''
    triplet_suffix = f'_triplet{args.triplet_alpha}' if args.triplet_alpha > 0 else ''
    seed_suffix = f'_seed{args.seed}' if args.seed != 69 else ''
    exp_name = (f'{args.model}{shot_suffix}{mixup_suffix}{ls_suffix}'
                f'{aug_suffix}{prompt_suffix}{triplet_suffix}{seed_suffix}')
    log_path = f'results/logs/{exp_name}_training_log.csv'
    checkpoint_path = f'results/checkpoints/best_{exp_name}.pth'

    best_val_acc = 0.0

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

        for epoch in range(args.epochs):
            start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                triplet_alpha=args.triplet_alpha, triplet_margin=args.triplet_margin)
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
