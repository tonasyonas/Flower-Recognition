# evaluate_cnn.py
"""
Single-model evaluation on the Flowers 102 test set.
Reports accuracy, mean per-class accuracy, top-5 accuracy, classification report,
and saves a normalized confusion matrix.

Usage:
    python evaluate_cnn.py --model vpt_deep --checkpoint results/checkpoints/best_vpt_deep.pth
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
from tqdm import tqdm

from models import MODEL_REGISTRY
from utils.data_loader import get_dataloaders


def evaluate(model, dataloader, device):
    """Run inference and return preds, labels, logits."""
    model.eval()
    preds, labels, all_logits = [], [], []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device, non_blocking=True)
            logits, _ = model(inputs)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(targets.numpy())
            all_logits.append(logits.cpu().numpy())
    return np.array(preds), np.array(labels), np.concatenate(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the Flowers 102 test set")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint .pth file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_prompts', type=int, default=10,
                        help='Number of prompts (VPT models only)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results/logs', exist_ok=True)

    # Build model (VPT models accept num_prompts)
    is_vpt = args.model.startswith('vpt_')
    model_cls = MODEL_REGISTRY[args.model]
    if is_vpt:
        model = model_cls(num_classes=102, num_prompts=args.num_prompts).to(device)
    else:
        model = model_cls(num_classes=102).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    print(f"Loaded {args.model} from {args.checkpoint}")

    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    preds, labels, logits = evaluate(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    mpc = balanced_accuracy_score(labels, preds)
    top5 = np.argsort(logits, axis=1)[:, -5:]
    top5_acc = float(np.mean([labels[i] in top5[i] for i in range(len(labels))]))

    print(f"\nResults for {args.model}:")
    print(f"  Accuracy:               {acc*100:.2f}%")
    print(f"  Mean Per-Class Acc:     {mpc*100:.2f}%   (primary metric)")
    print(f"  Top-5 Accuracy:         {top5_acc*100:.2f}%")

    # Classification report
    report = classification_report(labels, preds)
    report_path = f'results/logs/{args.model}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(
            f"Model: {args.model}\n"
            f"Checkpoint: {args.checkpoint}\n"
            f"Accuracy: {acc:.4f}\n"
            f"Mean Per-Class Accuracy: {mpc:.4f}\n"
            f"Top-5 Accuracy: {top5_acc:.4f}\n\n{report}"
        )
    print(f"\nReport saved to {report_path}")

    # Normalized confusion matrix
    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'{args.model} -- Test Acc: {acc:.4f}, MPC: {mpc:.4f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion')
    ticks = list(range(0, 102, 10))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.tight_layout()
    cm_path = f'results/logs/{args.model}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == '__main__':
    main()
